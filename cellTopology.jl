module cellTopology

using ..conicUtils

using StaticArrays

export classifyCellEigenvalue
export classifyCellEigenvector

# we use the same struct for if we are comparing both vec and val as if we are just doing val.
# just vec is a separate struct.
# I am not happy that I have to extend this to the eigenvector manifold. >:(
struct cellTopologyEigenvalue
    vertexTypesEigenvalue::SArray{Tuple{3}, Int8}
    vertexTypesEigenvector::SArray{Tuple{3}, Int8}
    DPArray::MArray{Tuple{10}, Int8}
    DNArray::MArray{Tuple{10}, Int8}
    RPArray::MArray{Tuple{10}, Int8}
    RNArray::MArray{Tuple{10}, Int8}
    RPArrayVec::MArray{Tuple{3}, Int8} # stores number of intersections with each edge
    RNArrayVec::MArray{Tuple{3}, Int8}
end

struct cellTopologyEigenvector
    vertexTypes::SArray{Tuple{3}, Int8}
    RPArray::MArray{Tuple{3},Int8}
    RNArray::MArray{Tuple{3},Int8}
end

struct Intersection
    x::Float64
    y::Float64
    code::Int8
end

# override sort function for intersections
# sorts in cyclic order.
function Base.isless(first::Intersection, second::Intersection)
    if first.y >= 0
        if second.y >= 0
            if first.x == second.x
                if first.x >= 0
                    return first.y < second.y
                else
                    return first.y > second.y
                end
            else
                return first.x > second.x
            end
        else
            return true
        end
    else
        if second.y >= 0
            return false
        else
            if first.x == second.x
                if first.x >= 0
                    return first.y < second.y
                else
                    return first.y > second.y
                end
            else
                return first.x < second.x
            end
        end
    end
end

# intersection codes (eigenvalue)
const BLANK::Int8 = 0
const E1::Int8 = 1
const E2::Int8 = 2
const E3::Int8 = 3
const DPRP::Int8 = 4 # d positive r positive
const DPRN::Int8 = 5 # d positive r negative
const DNRP::Int8 = 6 # d negative r positive
const DNRN::Int8 = 7 # d negative r negative
const INTERNAL_ELLIPSE = 8
const STRAIGHT_ANGLES = 9
# unfortunate intersection codes that I added by necessity :((
const CORNER_13 = 9 # when one conic intersects a corner
const CORNER_12 = 10
const CORNER_23 = 11
const CORNER_13_Z = 12 # when a conic intersects a corner and is zero.
const CORNER_12_Z = 13
const CORNER_23_Z = 14
const E1Z = 15 # when the entire tensor is equal to 0 at some point alone an edge.
const E2Z = 16
const E3Z = 17


# for eigenvalue, corners do not count as intersections, because
# the intersection of a region with a corner is given by the vertex classification.

# used for specifying that certain intersections are invalid / do not count.
const NULL = -1

# vertex types (eigenvalue)
const DP::Int8 = 0
const DN::Int8 = 1
const RP::Int8 = 2
const RN::Int8 = 3
const S::Int8 = 4
const RPTrumped = 5 # used for detecting P vs N for vertex eigenvector
const RZTrumped = 6
const RNTrumped = 7
const DZ::Int8 = 8 # used for detecting degenerate intersections.
const RZ::Int8 = 9

# vertex types (eigenvector)
const RRP::Int8 = 10
const DegenRP::Int8 = 11
const SRP::Int8 = 12
const SYM::Int8 = 13
const SRN::Int8 = 14
const DegenRN::Int8 = 15
const RRN::Int8 = 16
const Z::Int8 = 17 # all zeros.

# function getCellEdgeFromPoint(root::Root)
#     if root.y == 0.0
#         if root.x == 0.0
#             return CORNER_13
#         elseif root.x == 1.0
#             return CORNER_12
#         elseif root.doubleRoot
#             return E1TANGENT
#         else
#             return E1
#         end
#     elseif root.x == 0.0
#         if root.y == 1.0
#             return CORNER_23
#         elseif root.doubleRoot
#             return E3TANGENT
#         else
#             return E3
#         end
#     elseif root.doubleRoot
#         return E2TANGENT
#     else
#         return E2
#     end
# end

function classifyTensorEigenvalue(d,r,s)
    if abs(d) >= abs(r) && abs(d) > s && !isRelativelyClose(abs(d), s)
        if d > 0
            return DP
        elseif d < 0
            return DN
        else
            return Z
        end
    elseif abs(r) >= abs(d) && abs(r) > s && !isRelativelyClose(abs(r),s)
        if r > 0
            return RP
        elseif r < 0
            return RN
        else
            return Z
        end
    else
        return S
    end
end

function classifyTensorEigenvector(r,s)
    if isClose(r, 0.0)
        if isClose(s, 0.0)
            return Z
        else
            return SYM
        end
    elseif r > 0.0
        if isRelativelyClose(r,s)
            return DegenRP
        elseif r > s
            return RRP
        else
            return SRP
        end
    else
        if isRelativelyClose(-r,s)
            return DegenRN
        elseif -r > s
            return RRN
        else
            return SRN
        end
    end
end

function is_inside_triangle(x::Float64,y::Float64)
    return x >= 0.0 && y >= 0.0 && y <= 1.0-x
end

function valid_intersection_at_edge(DConic::conicEquation, RConic::conicEquation, x::Float64, y::Float64, DX, DY, DH, RX, RY, RH)
    return ( !(isClose(x,0.0) && isClose(y,0.0)) && !(isClose(x,0.0) && isClose(y,1.0)) && !(isClose(x,1.0) && isClose(y,0.0)) ) &&
           (!isClose(y, 0.0) || (x in DX && x in RX && ((dot(tangentDerivative(DConic, x, y), (0.0,1.0)) == 0.0) ⊻ (dot(tangentDerivative(RConic,x, y), (0.0,1.0)) == 0.0)))) &&
           (!isClose(y, 1.0-x) || (x in DH && x in RH && ((dot(tangentDerivative(DConic, x, y), (-1.0,-1.0)) == 0.0) ⊻ (dot(tangentDerivative(RConic,x, y), (-1.0,-1.0)) == 0.0)))) &&
           (!isClose(x,0.0) || (y in DY && y in RY && ((dot(tangentDerivative(DConic, x, y), (1.0,0.0)) == 0.0) ⊻ (dot(tangentDerivative(RConic,x, y), (1.0,0.0)) == 0.0))))
end

# returns the signs of d and r when the |d|=s and |r|=s curves intersect.
function DRSignAt(d1::Float64, d2::Float64, d3::Float64, x::Float64, y::Float64, same_sign::Bool)
    d = (d2-d1)*x + (d3-d1)*y + d1

    if same_sign
        if d > 0
            return DPRP
        elseif d < 0
            return DNRN
        else
            return NULL
        end
    else
        if d > 0
            return DPRN
        elseif d < 0
            return DNRP
        else
            return NULL
        end
    end
end

function classifyEllipseCenter(d1::Float64, d2::Float64, d3::Float64, r1::Float64, r2::Float64, r3::Float64, x::Float64, y::Float64)
    d = (d2-d1)*x + (d3-d1)*y + d1
    r = (r2-r1)*x + (r3-r1)*y + r1

    if abs(d) > abs(r)
        if d > 0
            return DP
        else
            return DN
        end
    else
        if r > 0
            return RP
        else
            return RN
        end
    end
end

# measures whether d dominates r where |d|=s intersects the cell wall. If not, returns NULL. If so, returns whether d is + or - at the intersection
# inputs two d and r values, and an interpolation value t between them, according to td1 + (1-t)d2
# so for edge 1, we should have d1 = point 2, d2 = point 1, t = x
#    for edge 2, we should have d1 = point 3, d2 = point 2, t = x
#    for edge 3, we should have d1 = point 3, d2 = point 1, t = y
function DCellIntersection(d1::Float64, d2::Float64, r1::Float64, r2::Float64, t::Float64)
    d = d1 * t + d2 * (1-t)
    r = r1 * t + r2 * (1-t)
    if abs(r) > abs(d)
        return NULL
    elseif d > 0.0
        return DP
    elseif d == 0.0
        return DZ
    else
        return DN
    end
end

# measures whether d dominates r where |d|=s intersects the cell wall. If not, returns NULL. If so, returns whether d is + or - at the intersection
# inputs two d and r values, and an interpolation value t between them, according to td1 + (1-t)d2
# so for edge 1, we should have d1 = point 2, d2 = point 1, t = x
#    for edge 2, we should have d1 = point 3, d2 = point 2, t = x
#    for edge 3, we should have d1 = point 3, d2 = point 1, t = y
function RCellIntersection(d1::Float64, d2::Float64, r1::Float64, r2::Float64, t::Float64)
    d = d1 * t + d2 * (1-t)
    r = r1 * t + r2 * (1-t)
    if abs(d) > abs(r)
        if r > 0
            return RPTrumped
        elseif r == 0
            return RZTrumped
        else
            return RNTrumped
        end
    elseif r > 0.0
        return RP
    elseif r == 0.0
        return RZ
    else
        return RN
    end
end

function hyperbola_intersection(point::Tuple{Float64,Float64},center::Tuple{Float64,Float64},axis::Tuple{Float64,Float64},coef::Float64,code::Int8)
    return Intersection( coef*((point[1] - center[1]) * axis[1] + (point[2] - center[2]) * axis[2]), -1.0, code )
end

function ellipse_intersection(point::Tuple{Float64,Float64},center::Tuple{Float64,Float64},axis1::Tuple{Float64,Float64},axis2::Tuple{Float64,Float64},code::Int8)
    return Intersection( (point[1]-center[1])*axis1[1] + (point[2]-center[2])*axis1[2],
                         (point[1]-center[1])*axis2[1] + (point[2]-center[2])*axis2[2],
                         code )
end

# extremely verbose, but I need all of this to be essentially inlined, so that's why
# I made it into a macro.
# first seven are variables. Positive is true if we're working with positive and false otherwise.
macro orientHyperbolaAndPush(point,center,axis1,axis2,orientation,list,code,positive)
    if positive
        return :(
            if $(esc(point)) == $(esc(center))
                push!($(esc(list)), Intersection(0.0, -1.0, $(esc(code))) )
            elseif $(esc(orientation)) == 0
                if ($(esc(point))[1] - $(esc(center))[1]) * $(esc(axis2))[1] + ($(esc(point))[2] - $(esc(center))[2]) * $(esc(axis2))[2] > 0
                    $(esc(orientation)) = 1
                    push!($(esc(list)), hyperbola_intersection($(esc(point)), $(esc(center)), $(esc(axis1)), 1.0, $(esc(code))))
                else
                    $(esc(orientation)) = 2
                    push!($(esc(list)), hyperbola_intersection($(esc(point)), $(esc(center)), $(esc(axis1)), -1.0, $(esc(code))))
                end
            elseif $(esc(orientation)) == 1
                push!($(esc(list)), hyperbola_intersection($(esc(point)), $(esc(center)), $(esc(axis1)), 1.0, $(esc(code))))
            else
                push!($(esc(list)), hyperbola_intersection($(esc(point)), $(esc(center)), $(esc(axis1)), -1.0, $(esc(code))))    
            end
        )
    else
        return :(   # :(
            if $(esc(point)) == $(esc(center))
                push!($(esc(list)), Intersection(0.0, -1.0, $(esc(code))) )
            elseif $(esc(orientation)) == 0
                if ($(esc(point))[1] - $(esc(center))[1]) * $(esc(axis2))[1] + ($(esc(point))[2] - $(esc(center))[2]) * $(esc(axis2))[2] > 0
                    $(esc(orientation)) = 2
                    push!($(esc(list)), hyperbola_intersection($(esc(point)), $(esc(center)), $(esc(axis1)), 1.0, $(esc(code))))
                else
                    $(esc(orientation)) = 1
                    push!($(esc(list)), hyperbola_intersection($(esc(point)), $(esc(center)), $(esc(axis1)), -1.0, $(esc(code))))                    
                end
            elseif $(esc(orientation)) == 2
                push!($(esc(list)), hyperbola_intersection($(esc(point)), $(esc(center)), $(esc(axis1)), 1.0, $(esc(code))))
            else
                push!($(esc(list)), hyperbola_intersection($(esc(point)), $(esc(center)), $(esc(axis1)), -1.0, $(esc(code))))    
            end
        )
    end
end

# everything is a variable except for d_positive and r_positive, which are set by the macro above.
macro placeIntersectionPointInLists(point,d_conic,r_conic,d_center,r_center,DVector1,DVector2,RVector1,RVector2,DList,RList,d_ellipse,r_ellipse,d_orientation,r_orientation,code,d_positive,r_positive)
    return :(begin

    dGrad = normalizedGradient($(esc(d_conic)), $(esc(point))[1], $(esc(point))[2])
    rGrad = normalizedGradient($(esc(r_conic)), $(esc(point))[1], $(esc(point))[2])
    dTangentVec = tangentDerivative(dGrad)
    rTangentVec = tangentDerivative(rGrad)

    if $(esc(d_ellipse))
        push!($(esc(DList)), ellipse_intersection($(esc(point)), $(esc(d_center)), $(esc(DVector1)), $(esc(DVector2)), Int8(sign(dot(dTangentVec,rGrad))) * $(esc(code))))
    else
        @orientHyperbolaAndPush($(esc(point)),$(esc(d_center)),$(esc(DVector1)),$(esc(DVector2)),$(esc(d_orientation)),$(esc(DList)), Int8(sign(dot(dTangentVec,rGrad))) * $(esc(code)),$d_positive)
    end

    if $(esc(r_ellipse))
        push!($(esc(RList)), ellipse_intersection($(esc(point)), $(esc(r_center)), $(esc(RVector1)), $(esc(RVector2)), Int8(sign(dot(rTangentVec,dGrad))) * $(esc(code))))
    else
        @orientHyperbolaAndPush($(esc(point)),$(esc(r_center)),$(esc(RVector1)),$(esc(RVector2)),$(esc(r_orientation)),$(esc(RList)), Int8(sign(dot(rTangentVec,dGrad))) * $(esc(code)),$r_positive)
    end

    end)
end

# likewise, we define checking intersection points as a macro in order to reduce huge amounts of repeated code
# all are variables.
macro checkIntersectionPoint(point,d_conic,r_conic,d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPList,DNList,RPList,RNList,d_ellipse,r_ellipse,d_orientation,r_orientation,code)
    return :(
    if $(esc(code)) == DPRP
        @placeIntersectionPointInLists($(esc(point)), $(esc(d_conic)), $(esc(r_conic)), $(esc(d_center)), $(esc(r_center)), $(esc(DVector1)), $(esc(DVector2)), $(esc(RVector1)), $(esc(RVector2)), $(esc(DPList)), $(esc(RPList)), $(esc(d_ellipse)), $(esc(r_ellipse)), $(esc(d_orientation)), $(esc(r_orientation)), $(esc(code)), true, true)
    elseif $(esc(code)) == DPRN
        @placeIntersectionPointInLists($(esc(point)), $(esc(d_conic)), $(esc(r_conic)), $(esc(d_center)), $(esc(r_center)), $(esc(DVector1)), $(esc(DVector2)), $(esc(RVector1)), $(esc(RVector2)), $(esc(DPList)), $(esc(RNList)), $(esc(d_ellipse)), $(esc(r_ellipse)), $(esc(d_orientation)), $(esc(r_orientation)), $(esc(code)), true, false)
    elseif $(esc(code)) == DNRP
        @placeIntersectionPointInLists($(esc(point)), $(esc(d_conic)), $(esc(r_conic)), $(esc(d_center)), $(esc(r_center)), $(esc(DVector1)), $(esc(DVector2)), $(esc(RVector1)), $(esc(RVector2)), $(esc(DNList)), $(esc(RPList)), $(esc(d_ellipse)), $(esc(r_ellipse)), $(esc(d_orientation)), $(esc(r_orientation)), $(esc(code)), false, true)
    else
        @placeIntersectionPointInLists($(esc(point)), $(esc(d_conic)), $(esc(r_conic)), $(esc(d_center)), $(esc(r_center)), $(esc(DVector1)), $(esc(DVector2)), $(esc(RVector1)), $(esc(RVector2)), $(esc(DNList)), $(esc(RNList)), $(esc(d_ellipse)), $(esc(r_ellipse)), $(esc(d_orientation)), $(esc(r_orientation)), $(esc(code)), false, false)
    end
    )
end

# negative means leaving, positive means entering
macro pushCodeFromSign(PList, NList, point, crossingCode, signCode, positiveTest, negativeTest, entering)
    return :(
        if $(esc(signCode)) == $positiveTest
            push!($(esc(PList)), Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode))
        elseif $(esc(signCode)) == $negativeTest
            push!($(esc(NList)), Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode))
        end
    )
end

macro pushCodeFromSignZero(PList, NList, point, crossingCode, crossingCodeZero, signCode, positiveTest, negativeTest, zeroTest, entering)
    return :(
        if $(esc(signCode)) == $positiveTest
            push!($(esc(PList)), Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode))
        elseif $(esc(signCode)) == $negativeTest
            push!($(esc(NList)), Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode))
        elseif $(esc(signCode)) == $zeroTest
            push!($(esc(PList)), Intersection($(esc(point))[1], $(esc(point))[2], $crossingCodeZero))
            push!($(esc(NList)), Intersection($(esc(point))[1], $(esc(point))[2], $crossingCodeZero))
        end
    )
end

function dot(a::Tuple{Float64,Float64},b::Tuple{Float64,Float64})
    return a[1]*b[1] + a[2]*b[2]
end

# returns a tuple of bools for whether, assuming that the two conic equations cross a boundary defined by vector axis at point, where inside points inside the triangle,
# does the conic equation defined by eq1 cross into the boundary or not
function doesConicEquationCrossDoubleBoundary(eq1::conicEquation, eq2::conicEquation, point::Tuple{Float64,Float64}, axis::Tuple{Float64,Float64}, inside::Tuple{Float64,Float64}, tangentVector::Tuple{Float64, Float64}, d::Bool)
    tangentVector2 = tangentDerivative(eq2, point[1], point[2])
    if tangentVector != tangentVector2
        d1DotInside = dot(tangentVector, inside)
        if isClose(d1DotInside, 0.0)
            return dot(gradient(eq2,point[1],point[2]),axis) < 0
        else
            d2DotInside = dot(tangentVector2, inside)
            if d2DotInside == 0.0
                return dot(normalizedGradient(eq2,point[1],point[2]),axis) < 0
            else
                grad = normalizedGradient(eq2, point[1], point[2])
                if sign(dot(tangentVector,inside))*dot(tangentVector,axis) < sign(dot(tangentVector2,inside))*dot(tangentVector2,axis)
                    return dot(grad,axis) > 0
                else
                    return dot(grad,axis) < 0
                end
            end
        end
    else
        grad1 = normalizedGradient(eq1, point[1], point[2])
        grad2 = normalizedGradient(eq2, point[1], point[2])
        if dot(grad1, grad2) < 0
            return true
        else
            k1 = curvature(eq1, point[1], point[2])
            k2 = curvature(eq2, point[1], point[2])
            if k1 > k2
                return false
            elseif k1 == k2
                return d
            else 
                return true
            end
        end
    end
end

function doesConicEquationCrossCorner(eq::conicEquation, x::Float64, y::Float64, in1::Tuple{Float64,Float64}, in2::Tuple{Float64,Float64}, grad::Tuple{Float64,Float64}, tangentVector::Tuple{Float64,Float64})
    dot1 = dot(tangentVector, in1)
    if isClose(dot1, 0.0)
        k = curvature(eq, x, y)
        return dot(grad, in1) > 0.0 && !isClose(k, 0.0)
    else
        dot2 = dot(tangentVector, in2)
        if isClose(dot2, 0.0)
            k = curvature(eq, x, y)
            return dot(grad, in2) > 0.0 && !isClose(k, 0.0)
        else
            return sign(dot1) == sign(dot2)
        end
    end
end

# coordinate transform functions used in the macro below
function e1x(x)
    return x
end

function e1y(x)
    return 0.0
end

function e2x(x)
    return x
end

function e2y(x)
    return 1.0-x
end

function e3x(y)
    return 0.0
end

function e3y(y)
    return y
end

# updates the eigenvector list
macro eigenvector_push(eigenvectorP, eigenvectorN, E, code)
    return :(
        if $(esc(code)) == RP || $(esc(code)) == RPTrumped
            $(esc(eigenvectorP))[$E] += 1
        elseif $(esc(code)) == RZ || $(esc(code)) == RZTrumped
            $(esc(eigenvectorP))[$E] += 1
            $(esc(eigenvectorN))[$E] += 1
        elseif $(esc(code)) == RN || $(esc(code)) == RNTrumped
            $(esc(eigenvectorN))[$E] += 1
        end
    )
end

# yes this is absolutely horrendus but the alternative is to write this out six times which is even worse.
# writing one macro that works is far less glitch prone
macro process_intercepts(edge_number, is_d, intercepts, alt_list, class_fun, d1, d2, r1, r2, PIntercepts, NIntercepts, conic, alt_conic, check_low, check_high,
                         do_eigenvector, do_eigenvector_runtime, eigenvectorP, eigenvectorN, any_intercepts, ignore_other)

    if is_d
        P = DP
        N = DN
        Z = DZ
    else
        P = RP
        N = RN
        Z = RZ
    end

    if edge_number == 1
        x = e1x
        y = e1y
        edge_orientation = (1.0,0.0)
        edge_inside = (0.0,1.0)
        low_edge_inside = (1.0, 0.0)
        high_edge_inside = (-1.0 / sqrt(2),-1.0 / sqrt(2))
        E = E1
        EZ = E1Z
        CORNER_L = CORNER_13
        CORNER_L_Z = CORNER_13_Z
        CORNER_H = CORNER_12
        CORNER_H_Z = CORNER_12_Z
        low_coords = (0.0,0.0)
        high_coords = (1.0,0.0)
    elseif edge_number == 2
        x = e2x
        y = e2y
        edge_orientation = (1.0 / sqrt(2),-1.0 / sqrt(2))
        edge_inside = (-1.0 / sqrt(2),-1.0 / sqrt(2))
        low_edge_inside = (1.0,0.0)
        high_edge_inside = (0.0,1.0)
        E = E2
        EZ = E2Z
        CORNER_L = CORNER_23
        CORNER_L_Z = CORNER_23_Z
        CORNER_H = CORNER_12
        CORNER_H_Z = CORNER_12_Z
        low_coords = (0.0,1.0)
        high_coords = (1.0,0.0)
    else
        x = e3x
        y = e3y
        edge_orientation = (0.0,1.0)
        edge_inside = (1.0,0.0)
        low_edge_inside = (0.0,1.0)
        high_edge_inside = (-1.0 / sqrt(2),-1.0 / sqrt(2))
        E = E3
        EZ = E3Z
        CORNER_L = CORNER_13
        CORNER_L_Z = CORNER_13_Z
        CORNER_H = CORNER_23
        CORNER_H_Z = CORNER_23_Z
        low_coords = (0.0,0.0)
        high_coords = (0.0,1.0)
    end

    return :(begin
    if -1e-10 <= $(esc(intercepts))[1] <= 1.0 + 1e-10
        $(esc(any_intercepts)) = true
        class = $(class_fun)($(esc(d1)), $(esc(d2)), $(esc(r1)), $(esc(r2)), $(esc(intercepts))[1])
        grad = normalizedGradient( $(esc(conic)), $x($(esc(intercepts))[1]), $y($(esc(intercepts))[1]) )
        tangentVector = tangentDerivative(grad)

        if isClose($(esc(intercepts))[1],0.0)
            if $check_low && (class == $Z || (
            doesConicEquationCrossCorner($(esc(conic)), ($low_coords)[1], $(low_coords)[2], $low_edge_inside, $edge_inside, grad, tangentVector) &&
            ($(esc(ignore_other)) || (!isClose($(esc(alt_list))[1],$(esc(intercepts))[1]) && !isClose($(esc(alt_list))[2],$(esc(intercepts))[1])) || 
                doesConicEquationCrossDoubleBoundary($(esc(conic)), $(esc(alt_conic)), ($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1])), $edge_orientation, $edge_inside, tangentVector, $is_d)
            )))

                entering = dot(tangentVector, $edge_inside)
                if isClose(entering, 0.0)
                    entering = dot(tangentVector, $low_edge_inside)
                end

                @pushCodeFromSignZero($(esc(PIntercepts)), $(esc(NIntercepts)), $low_coords, $CORNER_L, $CORNER_L_Z, class, $P, $N, $Z, sign(entering))

            end

        elseif isClose($(esc(intercepts))[1],1.0)
            if $check_high && (class == $Z || (
            doesConicEquationCrossCorner($(esc(conic)), $(high_coords)[1], $(high_coords)[2], $edge_inside, $high_edge_inside, grad, tangentVector) && 
            ($(esc(ignore_other)) || (!isClose($(esc(alt_list))[1],$(esc(intercepts))[1]) && !isClose($(esc(alt_list))[2],$(esc(intercepts))[1])) || 
                doesConicEquationCrossDoubleBoundary($(esc(conic)), $(esc(alt_conic)), ($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1])), (-$edge_orientation[1], -$edge_orientation[2]), $edge_inside, tangentVector, $is_d)
            )))
            
                entering = dot(tangentVector, $edge_inside)
                if isClose(entering, 0.0)
                    entering = dot(tangentVector, $high_edge_inside)
                end

                @pushCodeFromSignZero($(esc(PIntercepts)), $(esc(NIntercepts)), $high_coords, $CORNER_H, $CORNER_H_Z, class, $P, $N, $Z, sign(entering))
            end

        elseif isClose(dot(tangentVector,$edge_inside ),0.0) || isnan(tangentVector[1]) # e.g. if we have a non-transverse intersection
            if class == $Z
                push!($(esc(PIntercepts)), Intersection($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1]), $EZ))
                push!($(esc(NIntercepts)), Intersection($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1]), $EZ))
            end
        else
            if (!isClose($(esc(alt_list))[1],$(esc(intercepts))[1]) && !isClose($(esc(alt_list))[2],$(esc(intercepts))[1])) || isClose($(esc(alt_list))[1],$(esc(alt_list))[2]) || 
                doesConicEquationCrossDoubleBoundary($(esc(conic)), $(esc(alt_conic)), ($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1])), $edge_orientation, $edge_inside, tangentVector, $is_d)

                entering = dot(tangentVector, $edge_inside)

                @pushCodeFromSign($(esc(PIntercepts)), $(esc(NIntercepts)), ($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1])), $E, class, $P, $N, sign(entering))

            end
            if $do_eigenvector && $(esc(do_eigenvector_runtime))

                @eigenvector_push($(esc(eigenvectorP)), $(esc(eigenvectorN)), $E, class)

            end
        end
    end

    if -1e-10 <= $(esc(intercepts))[2] <= 1.0 + 1e-10 && $(esc(intercepts))[1] != $(esc(intercepts))[2]
        $(esc(any_intercepts)) = true
        class = $(class_fun)($(esc(d1)), $(esc(d2)), $(esc(r1)), $(esc(r2)), $(esc(intercepts))[2])
        grad = normalizedGradient( $(esc(conic)), $x($(esc(intercepts))[2]), $y($(esc(intercepts))[2]) )
        tangentVector = tangentDerivative(grad)        
        if isClose($(esc(intercepts))[2],0.0)
            if $check_low && (class == $Z || (
            doesConicEquationCrossCorner($(esc(conic)), ($low_coords)[1], $(low_coords)[2], $low_edge_inside, $edge_inside, grad, tangentVector) &&
            ($(esc(ignore_other)) || (!isClose($(esc(alt_list))[1],$(esc(intercepts))[2]) && !isClose($(esc(alt_list))[2],$(esc(intercepts))[2])) || 
                doesConicEquationCrossDoubleBoundary($(esc(conic)), $(esc(alt_conic)), ($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2])), $edge_orientation, $edge_inside, tangentVector, $is_d)
            )))
            
                entering = dot(tangentVector, $edge_inside)
                if isClose(entering, 0.0)
                    entering = dot(tangentVector, $low_edge_inside)
                end

                @pushCodeFromSignZero($(esc(PIntercepts)), $(esc(NIntercepts)), $low_coords, $CORNER_L, $CORNER_L_Z, class, $P, $N, $Z, sign(entering))

            end

        elseif isClose($(esc(intercepts))[2],1.0) || isnan(tangentVector[1])
            if $check_high && (class == $Z || (
            doesConicEquationCrossCorner($(esc(conic)), $(high_coords)[1], $(high_coords)[2], $edge_inside, $high_edge_inside, grad, tangentVector) && 
            ($(esc(ignore_other)) || (!isClose($(esc(alt_list))[1],$(esc(intercepts))[2]) && !isClose($(esc(alt_list))[2],$(esc(intercepts))[2])) || 
                doesConicEquationCrossDoubleBoundary($(esc(conic)), $(esc(alt_conic)), ($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2])), (-$edge_orientation[1], -$edge_orientation[2]), $edge_inside, tangentVector, $is_d)
            )))

                entering = dot(tangentVector, $edge_inside)
                if isClose(entering, 0.0)
                    entering = dot(tangentVector, $high_edge_inside)
                end

                @pushCodeFromSignZero($(esc(PIntercepts)), $(esc(NIntercepts)), $high_coords, $CORNER_H, $CORNER_H_Z, class, $P, $N, $Z, sign(entering))

            end
        elseif isClose(dot(tangentVector,$edge_inside ),0.0) # e.g. if we have a non-transverse intersection
            if class == $Z         
                push!($(esc(PIntercepts)), Intersection($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2]), $EZ))
                push!($(esc(NIntercepts)), Intersection($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2]), $EZ))
            end
        else               
            if (!isClose($(esc(alt_list))[1],$(esc(intercepts))[2]) && !isClose($(esc(alt_list))[2],$(esc(intercepts))[2])) || isClose($(esc(alt_list))[1],$(esc(alt_list))[2]) || 
                doesConicEquationCrossDoubleBoundary($(esc(conic)), $(esc(alt_conic)), ($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2])), $edge_orientation, $edge_inside, tangentVector, $is_d)

                entering = dot(tangentVector, $edge_inside)

                @pushCodeFromSign($(esc(PIntercepts)), $(esc(NIntercepts)), ($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2])), $E, class, $P, $N, sign(entering))
            end
            if $do_eigenvector && $(esc(do_eigenvector_runtime))
                @eigenvector_push($(esc(eigenvectorP)), $(esc(eigenvectorN)), $E, class)
            end
        end
    end
    end)
end

function getAxesAndCheckIfSortAsEllipse(eq::conicEquation, class::Int64)
    if class == ELLIPSE
        if eq.B == 0
            return ((1.0,0.0), (0.0,-1.0), true)
        else
            eigenRoot = sqrt( 4 * eq.B^2 + ( 2*eq.A - 2*eq.C )^2 )
            λ2 = (2*eq.A + 2*eq.C - eigenRoot) / 2 # we only need the second (negative) eigenvalue.
            axis1x = (λ2 - 2*eq.C) / eq.B

            if axis1x > 0
                # double check to make sure that this actually always gives the first has positive slope and the second has negative
                vector1 = (axis1x, 1.0)
                vector2 = (1.0/axis1x, -1.0)
            else
                vector1 = (-1.0/axis1x, 1.0)
                vector2 = (-axis1x, -1.0)
            end

            return (vector1, vector2, true)
        end
    elseif class == HYPERBOLA || class == INTERSECTING_LINES
        if eq.B == 0
            disc = sign(eq.F - eq.D^2 / (4 * eq.A) - eq.E^2 / (4 * eq.C))
            if disc == sign(eq.A) || disc == 0 && eq.A < 0 
                return ((-1.0,0.0), (0.0,1.0), false)
            else
                return ((0.0,1.0), (1.0,0.0), false)
            end
        else
            eigenRoot = sqrt( 4 * eq.B^2 + ( 2*eq.A - 2*eq.C )^2 )
            λ2 = (2*eq.A + 2*eq.C - eigenRoot) / 2 # we only need the second (negative) eigenvalue.
            axis1x = (λ2 - 2*eq.C) / eq.B

            if axis1x > 0.0
                vector1 = (-axis1x, -1.0)
                vector2 = (-1.0/axis1x, 1.0)
            else
                vector1 = (axis1x, 1.0)
                vector2 = (-1.0/axis1x, 1.0)
            end

            return (vector1, vector2, false)
        end
    elseif class == PARABOLA || class == PARALLEL_INNER
        if eq.B == 0
            return ((1.0,0.0), (0.0,-1.0), true)
        else
            axis1x = 2*eq.A / eq.B

            if axis1x > 0
                # double check to make sure that this actually always gives the first has positive slope and the second has negative
                vector1 = (axis1x, 1.0)
                vector2 = (1.0/axis1x, -1.0)
            else
                vector1 = (-1.0/axis1x, 1.0)
                vector2 = (-axis1x, -1.0)
            end
            
            return (vector1, vector2, true)
        end
    elseif class == PARALLEL_OUTER
        axis1x = - eq.B / (2*eq.A)

        if axis1x > 0.0
            vector1 = (-axis1x, -1.0)
            vector2 = (-1.0/axis1x, 1.0)
        else
            vector1 = (axis1x, 1.0)
            vector2 = (-1.0/axis1x, 1.0)
        end

        return (vector1, vector2, false)        
    elseif class == SIDEWAYS_PARABOLA || class == PARALLEL_INNER_HORIZONTAL
        return ((1.0,0.0), (0.0,-1.0), true)
    elseif class == PARALLEL_OUTER_HORIZONTAL || class == HORIZONTAL_LINE
        return ((-1.0,0.0),(0.0,1.0), false)
    elseif class == LINE
        return ((-1.0,eq.D/eq.E), (eq.E/eq.D,1.0), false)
    elseif class == VERTICAL_LINE
        return ((0.0,1.0), (1.0,0.0), false)
    else
        return ((0.0,0.0),(0.0,0.0),false) # I don't think this case will ever be reached, but it is here for type stability.
    end
end

function isClose(x1::Float64, x2::Float64)
    return abs(x1-x2) < 1e-10
end

function isGreater(x1::Float64, x2::Float64)
    return x1 > x2 + 1e-10
end

function isLess(x1::Float64, x2::Float64)
    return x1 < x2 - 1e-10
end

function isRelativelyClose(x1::Float64, x2::Float64)
    return abs(x1-x2) < 1e-10 * max(x1,x2)
end

# While technically we use a barycentric interpolation scheme which is agnostic to the locations of the actual cell vertices,
# for mathematical ease we assume that point 1 is at (0,0), point 2 is at (1,0), and point 3 is at (0,1). Choosing a specific
# embedding will not affect the topology.
# The final bool tells whether we simultaneously compute eigenvector topology.
function classifyCellEigenvalue(M1::SMatrix{2,2,Float64}, M2::SMatrix{2,2,Float64}, M3::SMatrix{2,2,Float64},eigenvector::Bool,verbose::Bool=false)
    DPArray = MArray{Tuple{10}, Int8}(zeros(Int8, 10))
    DNArray = MArray{Tuple{10}, Int8}(zeros(Int8, 10))
    RPArray = MArray{Tuple{10}, Int8}(zeros(Int8, 10))
    RNArray = MArray{Tuple{10}, Int8}(zeros(Int8, 10))
    RPArrayVec = MArray{Tuple{3}, Int8}(zeros(Int8, 3))
    RNArrayVec = MArray{Tuple{3}, Int8}(zeros(Int8, 3))

    # we work with everything *2. It does not change the results.

    d1 = M1[1,1]+M1[2,2]
    r1 = M1[2,1]-M1[1,2]
    cos1 = M1[1,1]-M1[2,2]
    sin1 = M1[1,2]+M1[2,1]
    s1 = sqrt(cos1^2+sin1^2)

    d2 = M2[1,1]+M2[2,2]
    r2 = M2[2,1]-M2[1,2]
    cos2 = M2[1,1]-M2[2,2]
    sin2 = M2[1,2]+M2[2,1]
    s2 = sqrt(cos2^2+sin2^2)

    d3 = M3[1,1]+M3[2,2]
    r3 = M3[2,1]-M3[1,2]
    cos3 = M3[1,1]-M3[2,2]
    sin3 = M3[1,2]+M3[2,1]
    s3 = sqrt(cos3^2+sin3^2)

    vertexTypesEigenvalue = SArray{Tuple{3},Int8}((classifyTensorEigenvalue(d1,r1,s1), classifyTensorEigenvalue(d2,r2,s2), classifyTensorEigenvalue(d3,r3,s3)))

    if eigenvector
        vertexTypesEigenvector = SArray{Tuple{3},Int8}((classifyTensorEigenvector(r1,s1), classifyTensorEigenvector(r2,s2), classifyTensorEigenvector(r3, s3)))
    else
        vertexTypesEigenvector = SArray{Tuple{3},Int8}((0,0,0))
    end

    if (vertexTypesEigenvalue[1] == Z && vertexTypesEigenvalue[2] == Z && vertexTypesEigenvalue[3] == Z) ||
       ( !eigenvector && isGreater(abs(d1),s1) && isGreater(abs(d2),s2) && isGreater(abs(d3),s3) && ( ( isGreater(d1,0.0) && isGreater(d2,0.0) && isGreater(d3,0.0)) || ( isLess(d1,0.0) && isLess(d2,0.0) && isLess(d3,0.0) ) ) ) ||
       ( isGreater(abs(r1),s1) && isGreater(abs(r2),s2) && isGreater(abs(r3),s3) && ( ( isGreater(r1,0.0) && isGreater(r2,0.0) && isGreater(r3,0.0) ) || ( isLess(r1,0.0) && isLess(r2,0.0) && isLess(r3,0.0) ) ) ) ||
       (isClose(s1,0.0) && isClose(s2,0.0) && isClose(s3,0.0))
       # in this case, s is dominated by d or r throughout the entire triangle, so the topology follows from the vertices.
        return cellTopologyEigenvalue(vertexTypesEigenvalue, vertexTypesEigenvector, DPArray, DNArray, RPArray, RNArray, RPArrayVec, RNArrayVec)
    end

    # generate conics and intersect them with the triangles:

    DBase = interpolationConic(d1, d2, d3)
    RBase = interpolationConic(r1, r2, r3)
    sinBase = interpolationConic(sin1, sin2, sin3)
    cosBase = interpolationConic(cos1, cos2, cos3)
    sinPlusCos = add(sinBase, cosBase)

    DConic = sub(DBase, sinPlusCos)
    RConic = sub(RBase, sinPlusCos)

    DConicXIntercepts = quadraticFormula(DConic.A, DConic.D, DConic.F) # gives x coordinate
    DConicYIntercepts = quadraticFormula(DConic.C, DConic.E, DConic.F) # gives y coordinate
    # hypotenuse intercepts. Gives x coordinate
    DConicHIntercepts = quadraticFormula(DConic.A - DConic.B + DConic.C, DConic.B - 2*DConic.C + DConic.D - DConic.E, DConic.C + DConic.E + DConic.F)

    RConicXIntercepts = quadraticFormula(RConic.A, RConic.D, RConic.F) # gives x coordinate
    RConicYIntercepts = quadraticFormula(RConic.C, RConic.E, RConic.F) # gives y coordinate
    # hypotenuse intercepts. Gives x coordinate
    RConicHIntercepts = quadraticFormula(RConic.A - RConic.B + RConic.C, RConic.B - 2*RConic.C + RConic.D - RConic.E, RConic.C + RConic.E + RConic.F)

    # first, check for degenerate cases of (a) single line with no intersection region, and (b) single point.
    # in either of these cases, we completely ignore the given conic.
    d_type,d_center = classifyAndReturnCenter(DConic)
    r_type,r_center = classifyAndReturnCenter(RConic)

    ignore_d = (d_type == POINT || d_type == EMPTY)
    ignore_r = (r_type == POINT || r_type == EMPTY)

    # third elt is true for double root, false otherwise
    DPIntercepts = Vector{Intersection}([])
    DNIntercepts = Vector{Intersection}([])
    RPIntercepts = Vector{Intersection}([])
    RNIntercepts = Vector{Intersection}([])

    sizehint!(DPIntercepts, 6)
    sizehint!(RPIntercepts, 6)
    sizehint!(DPIntercepts, 6)
    sizehint!(RNIntercepts, 6)

    # at this point the loop unrolling was so bad that we had to outsource to macros. It is truly awful.

    # process D intercepts

    any_d_intercepts = false
    any_r_intercepts = false

    if !ignore_d
        # process D intercepts
        # (edge_number, is_d, intercepts, alt_list, class_fun, d1, d2, r1, r2, PIntercepts, NIntercepts, conic, alt_conic, check_low, check_high,
        # do_eigenvector, do_eigenvector_runtime, eigenvector_c, eigenvectorP, eigenvectorN, any_intercepts, ignore_other)

        @process_intercepts(1, true, DConicXIntercepts, RConicXIntercepts, DCellIntersection, d2, d1, r2, r1, DPIntercepts, DNIntercepts, DConic, RConic, true, true, 
        false, false, RPArrayVec, RNArrayVec, any_d_intercepts, ignore_r)

        @process_intercepts(2, true, DConicHIntercepts, RConicHIntercepts, DCellIntersection, d2, d3, r2, r3, DPIntercepts, DNIntercepts, DConic, RConic, false, false, 
        false, false, RPArrayVec, RNArrayVec, any_d_intercepts, ignore_r)

        @process_intercepts(3, true, DConicYIntercepts, RConicYIntercepts, DCellIntersection, d3, d1, r3, r1, DPIntercepts, DNIntercepts, DConic, RConic, false, true, 
        false, false, RPArrayVec, RNArrayVec, any_d_intercepts, ignore_r)
    end

    if !ignore_r
        # process R intercepts
        @process_intercepts(1, false, RConicXIntercepts, DConicXIntercepts, RCellIntersection, d2, d1, r2, r1, RPIntercepts, RNIntercepts, RConic, DConic, true, true, 
        true, eigenvector, RPArrayVec, RNArrayVec, any_r_intercepts, ignore_d)

        @process_intercepts(2, false, RConicHIntercepts, DConicHIntercepts, RCellIntersection, d2, d3, r2, r3, RPIntercepts, RNIntercepts, RConic, DConic, false, false, 
        true, eigenvector, RPArrayVec, RNArrayVec, any_r_intercepts, ignore_d)

        @process_intercepts(3, false, RConicYIntercepts, DConicYIntercepts, RCellIntersection, d3, d1, r3, r1, RPIntercepts, RNIntercepts, RConic, DConic, false, true, 
        true, eigenvector, RPArrayVec, RNArrayVec, any_r_intercepts, ignore_d)
    end

    # Check if either is an internal ellipse
    # Check that each conic (a) does not not intersect the triangle, (b) is an ellipse, and (c) has a center inside the triangle.
    # Checking whether or not the conic is an ellipse follows from the sign of the discriminant.
    d_internal_ellipse = false
    r_internal_ellipse = false

    if d_type == ELLIPSE && !any_d_intercepts && (!eigenvector || !(abs(d1) >= s1 && abs(d2) >= s2 && abs(d3) >= s3) ) && is_inside_triangle(d_center[1], d_center[2]) && !ignore_d
        d_internal_ellipse = true
    end

    if r_type == ELLIPSE && !any_r_intercepts && is_inside_triangle(r_center[1], r_center[2]) && !ignore_r
        r_internal_ellipse = true
    end

    # if the two conics do not intersect the triangle, and neither is an internal ellipse,
    # then the triangle is a standard white triangle.

    if !any_d_intercepts && !any_r_intercepts && !d_internal_ellipse && !r_internal_ellipse
        return cellTopologyEigenvalue(vertexTypesEigenvalue, vertexTypesEigenvector, DPArray, DNArray, RPArray, RNArray, RPArrayVec, RNArrayVec)
    end

    # If we made it this far, it means that this is an "interesting" case :(

    # So now we need to intersect the conics with each other. We can find the intersection points
    # where each individual conic intersects the lines r=d and r=-d.

    DPPoints = Vector{Intersection}(undef, 0)
    DNPoints = Vector{Intersection}(undef, 0)
    RPPoints = Vector{Intersection}(undef, 0)
    RNPoints = Vector{Intersection}(undef, 0)

    sizehint!(DPPoints, 10)
    sizehint!(DNPoints, 10)
    sizehint!(RPPoints, 10)
    sizehint!(RNPoints, 10)

    DVector1, DVector2, d_ellipse = getAxesAndCheckIfSortAsEllipse(DConic, d_type)
    RVector1, RVector2, r_ellipse = getAxesAndCheckIfSortAsEllipse(RConic, r_type)

    d_orientation = 0
    r_orientation = 0

    # add edge intersections to list using axis transform

    if d_ellipse
        for i in eachindex(DPIntercepts)
            push!(DPPoints, ellipse_intersection((DPIntercepts[i].x, DPIntercepts[i].y), d_center, DVector1, DVector2, DPIntercepts[i].code))
        end

        for i in eachindex(DNIntercepts)
            push!(DNPoints, ellipse_intersection((DNIntercepts[i].x, DNIntercepts[i].y), d_center, DVector1, DVector2, DNIntercepts[i].code))
        end
    else
        # verbose, but I need to reduce the number of checks that happen
        for i in eachindex(DPIntercepts)
            @orientHyperbolaAndPush((DPIntercepts[i].x,DPIntercepts[i].y),d_center,DVector1,DVector2,d_orientation,DPPoints,DPIntercepts[i].code,true)
        end

        for i in eachindex(DNIntercepts)
            @orientHyperbolaAndPush((DNIntercepts[i].x,DNIntercepts[i].y),d_center,DVector1,DVector2,d_orientation,DNPoints,DNIntercepts[i].code,false)
        end
    end

    if r_ellipse
        for i in eachindex(RPIntercepts)
            push!(RPPoints, ellipse_intersection((RPIntercepts[i].x, RPIntercepts[i].y), r_center, RVector1, RVector2, RPIntercepts[i].code))
        end

        for i in eachindex(RNIntercepts)
            push!(RNPoints, ellipse_intersection((RNIntercepts[i].x, RNIntercepts[i].y), r_center, RVector1, RVector2, RNIntercepts[i].code))
        end
    else
        # verbose, but I need to reduce the number of checks that happen
        for i in eachindex(RPIntercepts)
            @orientHyperbolaAndPush((RPIntercepts[i].x,RPIntercepts[i].y),r_center,RVector1,RVector2,r_orientation,RPPoints,RPIntercepts[i].code,true)
        end

        for i in eachindex(RNIntercepts)
            @orientHyperbolaAndPush((RNIntercepts[i].x,RNIntercepts[i].y),r_center,RVector1,RVector2,r_orientation,RNPoints,RNIntercepts[i].code,false)
        end
    end

    if !ignore_d && !ignore_r

        # coefficients for standard form lines of r=0 and d=0 (in the form of ax+by+c=0)
        a_r = r2-r1
        b_r = r3-r1
        c_r = r1
        a_d = d2-d1
        b_d = d3-d1
        c_d = d1

        # coefficients for r=d in standard form
        a_rpd = a_r - a_d
        b_rpd = b_r - b_d
        c_rpd = c_r - c_d

        # coefficients for r=-d in standard form
        a_rnd = a_r + a_d
        b_rnd = b_r + b_d
        c_rnd = c_r + c_d

        # intersect r conic with both of these lines
        rpd_intersections = intersectWithStandardFormLine(RConic, a_rpd, b_rpd, c_rpd)
        rnd_intersections = intersectWithStandardFormLine(RConic, a_rnd, b_rnd, c_rnd)

        rpd_intersection_1 = NULL
        rpd_intersection_2 = NULL
        rnd_intersection_1 = NULL
        rnd_intersection_2 = NULL

        if rpd_intersections[1] != rpd_intersections[2] # non-transverse intersections are not counted
            if is_inside_triangle(rpd_intersections[1][1], rpd_intersections[1][2]) && valid_intersection_at_edge(DConic, RConic, rpd_intersections[1][1], rpd_intersections[1][2], DConicXIntercepts, DConicYIntercepts, DConicHIntercepts, RConicXIntercepts, RConicYIntercepts, RConicHIntercepts)
                rpd_intersection_1 = DRSignAt(d1, d2, d3, rpd_intersections[1][1], rpd_intersections[1][2], true)
            end

            if is_inside_triangle(rpd_intersections[2][1], rpd_intersections[2][2]) && valid_intersection_at_edge(DConic, RConic, rpd_intersections[2][1], rpd_intersections[2][2], DConicXIntercepts, DConicYIntercepts, DConicHIntercepts, RConicXIntercepts, RConicYIntercepts, RConicHIntercepts) 
                rpd_intersection_2 = DRSignAt(d1, d2, d3, rpd_intersections[2][1], rpd_intersections[2][2], true)
            end
        end

        if rnd_intersections[1] != rnd_intersections[2]
            if is_inside_triangle(rnd_intersections[1][1], rnd_intersections[1][2]) && valid_intersection_at_edge(DConic, RConic, rnd_intersections[1][1], rnd_intersections[1][2], DConicXIntercepts, DConicYIntercepts, DConicHIntercepts, RConicXIntercepts, RConicYIntercepts, RConicHIntercepts)
                rnd_intersection_1 = DRSignAt(d1, d2, d3, rnd_intersections[1][1], rnd_intersections[1][2], false)                
            end

            if is_inside_triangle(rnd_intersections[2][1], rnd_intersections[2][2])  && valid_intersection_at_edge(DConic, RConic, rnd_intersections[2][1], rnd_intersections[2][2], DConicXIntercepts, DConicYIntercepts, DConicHIntercepts, RConicXIntercepts, RConicYIntercepts, RConicHIntercepts)
                rnd_intersection_2 = DRSignAt(d1, d2, d3, rnd_intersections[2][1], rnd_intersections[2][2], false)
            end
        end

        # then check each of the four intersection points
        # using macros here actually saved ~200 lines of code (not an exaggeration)
        if rpd_intersection_1 != NULL
            @checkIntersectionPoint(rpd_intersections[1],DConic, RConic, d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPPoints,DNPoints,RPPoints,RNPoints,d_ellipse,r_ellipse,d_orientation,r_orientation,rpd_intersection_1)
        end

        if rpd_intersection_2 != NULL
            @checkIntersectionPoint(rpd_intersections[2],DConic, RConic, d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPPoints,DNPoints,RPPoints,RNPoints,d_ellipse,r_ellipse,d_orientation,r_orientation,rpd_intersection_2)
        end

        if rnd_intersection_1 != NULL
            @checkIntersectionPoint(rnd_intersections[1],DConic, RConic, d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPPoints,DNPoints,RPPoints,RNPoints,d_ellipse,r_ellipse,d_orientation,r_orientation,rnd_intersection_1)
        end

        if rnd_intersection_2 != NULL
            @checkIntersectionPoint(rnd_intersections[2],DConic, RConic, d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPPoints,DNPoints,RPPoints,RNPoints,d_ellipse,r_ellipse,d_orientation,r_orientation,rnd_intersection_2)
        end

    end

    # custom sort function that ensures the clockwise ordering that we were shooting for.
    sort!(DPPoints)
    sort!(DNPoints)
    sort!(RPPoints)
    sort!(RNPoints)

    for i in eachindex(DPPoints)
        DPArray[i] = DPPoints[i].code
    end

    for i in eachindex(DNPoints)
        DNArray[i] = DNPoints[i].code
    end

    for i in eachindex(RPPoints)
        RPArray[i] = RPPoints[i].code
    end

    for i in eachindex(RNPoints)
        RNArray[i] = RNPoints[i].code
    end
                                                                                                            # we only check d1 since either all are greater than s1 or all are less
    if d_type == ELLIPSE && !any_d_intercepts && length(DPPoints) == 0 && length(DNPoints) == 0 && is_inside_triangle(d_center[1], d_center[2]) && (!eigenvector || abs(d1) < s1)
        d_center_class = classifyEllipseCenter(d1, d2, d3, r1, r2, r3, d_center[1], d_center[2])
        if d_center_class == DP
            DPArray[length(DPPoints)+1] = INTERNAL_ELLIPSE
        elseif d_center_class == DN
            DNArray[length(DNPoints)+1] = INTERNAL_ELLIPSE
        end
    end

    if !any_r_intercepts && r_type == ELLIPSE && is_inside_triangle(r_center[1], r_center[2])
        if eigenvector
            if (r2-r1)*r_center[1]+(r3-r1)*r_center[2]+r1 >= 0
                RPArrayVec[1] = INTERNAL_ELLIPSE
            else
                RNArrayVec[1] = INTERNAL_ELLIPSE
            end
        end

        if length(RPPoints) == 0 && length(RNPoints) == 0
            r_center_class = classifyEllipseCenter(d1,d2,d3,r1,r2,r3,r_center[1],r_center[2])
            if r_center_class == RP
                RPArray[1] = INTERNAL_ELLIPSE
            elseif r_center_class == RN
                RNArray[1] = INTERNAL_ELLIPSE
            end
        end

    end

    if eigenvector 
        if vertexTypesEigenvector[1] == DegenRP || vertexTypesEigenvector[2] == DegenRP || vertexTypesEigenvector[3] == DegenRP
            if s1 != 0.0
                θ1 = asin(sin1/s1)
            else
                θ1 = 0.0
            end

            if s2 != 0.0
                θ2 = asin(sin2/s2)
            else
                θ2 = 0.0
            end

            if s3 != 0.0
                θ3 = asin(sin3/s3)
            else
                θ3 = 0.0
            end

            if isClose(abs(sin(θ1)), abs(sin(θ2))) && vertexTypesEigenvector[1] == DegenRP && vertexTypesEigenvector[2] == DegenRP
                RPArrayVec[1] = STRAIGHT_ANGLES
            end

            if isClose(abs(sin(θ2)), abs(sin(θ3))) && vertexTypesEigenvector[2] == DegenRP && vertexTypesEigenvector[3] == DegenRP
                RPArrayVec[2] = STRAIGHT_ANGLES
            end

            if isClose(abs(sin(θ3)), abs(sin(θ1))) && vertexTypesEigenvector[3] == DegenRP && vertexTypesEigenvector[1] == DegenRP
                RPArrayVec[3] = STRAIGHT_ANGLES
            end
        elseif vertexTypesEigenvector[1] == DegenRN || vertexTypesEigenvector[2] == DegenRN || vertexTypesEigenvector[3] == DegenRN
            if s1 != 0.0
                θ1 = asin(sin1/s1)
            else
                θ1 = 0.0
            end

            if s2 != 0.0
                θ2 = asin(sin2/s2)
            else
                θ2 = 0.0
            end

            if s3 != 0.0
                θ3 = asin(sin3/s3)
            else
                θ3 = 0.0
            end

            if isClose(abs(sin(θ1)), abs(sin(θ2))) && vertexTypesEigenvector[1] == DegenRN && vertexTypesEigenvector[2] == DegenRN
                RNArrayVec[1] = STRAIGHT_ANGLES
            end

            if isClose(abs(sin(θ2)), abs(sin(θ3))) && vertexTypesEigenvector[2] == DegenRN && vertexTypesEigenvector[3] == DegenRN
                RNArrayVec[2] = STRAIGHT_ANGLES
            end

            if isClose(abs(sin(θ3)), abs(sin(θ1))) && vertexTypesEigenvector[3] == DegenRN && vertexTypesEigenvector[1] == DegenRN
                RNArrayVec[3] = STRAIGHT_ANGLES
            end            
        end
    end

    # what a racket
    return cellTopologyEigenvalue(vertexTypesEigenvalue, vertexTypesEigenvector, DPArray, DNArray, RPArray, RNArray, RPArrayVec, RNArrayVec)
end

function classifyCellEigenvector(M1::SMatrix{2,2,Float64}, M2::SMatrix{2,2,Float64}, M3::SMatrix{2,2,Float64})
    RPArray = MArray{Tuple{3}, Int8}(zeros(Int8, 3))
    RNArray = MArray{Tuple{3}, Int8}(zeros(Int8, 3))

    # we work with everything *2. It does not change the results.

    r1 = M1[2,1]-M1[1,2]
    cos1 = M1[1,1]-M1[2,2]
    sin1 = M1[1,2]+M1[2,1]
    s1 = sqrt(cos1^2+sin1^2)

    r2 = M2[2,1]-M2[1,2]
    cos2 = M2[1,1]-M2[2,2]
    sin2 = M2[1,2]+M2[2,1]
    s2 = sqrt(cos2^2+sin2^2)

    r3 = M3[2,1]-M3[1,2]
    cos3 = M3[1,1]-M3[2,2]
    sin3 = M3[1,2]+M3[2,1]
    s3 = sqrt(cos3^2+sin3^2)

    vertexTypes = SArray{Tuple{3},Int8}((classifyTensorEigenvector(r1,s1), classifyTensorEigenvector(r2,s2), classifyTensorEigenvector(r3,s3)))

    if isGreater(abs(r1),s1) && isGreater(abs(r2),s2) && isGreater(abs(r3),s3) && ( ( isGreater(r1, 0.0) && isGreater(r2, 0.0) && isGreater(r3, 0.0) ) || ( isLess(r1, 0.0) && isLess(r2,0.0) && isLess(r3,0.0) ) )
       # in this case, s is dominated by d or r throughout the entire triangle, so the topology follows from the vertices.
        return cellTopologyEigenvector(vertexTypes, RPArray, RNArray)
    end

    # generate conics and intersect them with the triangles:

    RBase = interpolationConic(r1, r2, r3)
    sinBase = interpolationConic(sin1, sin2, sin3)
    cosBase = interpolationConic(cos1, cos2, cos3)
    sinPlusCos = add(sinBase, cosBase)

    RConic = sub(RBase, sinPlusCos)

    RConicXIntercepts = quadraticFormula(RConic.A, RConic.D, RConic.F) # gives x coordinate
    RConicYIntercepts = quadraticFormula(RConic.C, RConic.E, RConic.F) # gives y coordinate
    # hypotenuse intercepts. Gives x coordinate
    RConicHIntercepts = quadraticFormula(RConic.A - RConic.B + RConic.C, RConic.B - 2*RConic.C + RConic.D - RConic.E, RConic.C + RConic.E + RConic.F)

    if 1e-10 < RConicXIntercepts[1] < 1.0-1e-10 && !isClose(dot(normalizedGradient(RConic, RConicXIntercepts[1], 0.0), (0.0,1.0)), 0.0)
        r = RConicXIntercepts[1]*r2 + (1.0-RConicXIntercepts[1])*r1
        if isGreater(r, 0.0)
            RPArray[1] += 1
        elseif isClose(r, 0.0)
            RPArray[1] += 1            
            RNArray[1] += 1
        else
            RNArray[1] += 1
        end
    end

    if 1e-10 < RConicXIntercepts[2] < 1.0-1e-10 && !isClose(dot(normalizedGradient(RConic, RConicXIntercepts[2], 0.0), (0.0,1.0)), 0.0) &&
        RConicXIntercepts[1] != RConicXIntercepts[2]

        r = RConicXIntercepts[2]*r2 + (1.0-RConicXIntercepts[2])*r1
        if isGreater(r, 0.0)
            RPArray[1] += 1
        elseif isClose(r, 0.0)
            RPArray[1] += 1            
            RNArray[1] += 1
        else
            RNArray[1] += 1
        end
    end

    if 1e-10 < RConicYIntercepts[1] < 1.0-1e-10 && !isClose(dot(normalizedGradient(RConic, 0.0, RConicYIntercepts[1]), (1.0,0.0)), 0.0)

        r = RConicYIntercepts[2]*r3 + (1.0-RConicYIntercepts[2])*r1
        if isGreater(r, 0.0)
            RPArray[3] += 1
        elseif isClose(r, 0.0)
            RPArray[3] += 1            
            RNArray[3] += 1
        else
            RNArray[3] += 1
        end
    end

    if 1e-10 < RConicYIntercepts[2] < 1.0-1e-10 && !isClose(dot(normalizedGradient(RConic, 0.0, RConicYIntercepts[2]), (1.0,0.0)), 0.0) &&
        RConicYIntercepts[1] != RConicYIntercepts[2]

        r = RConicYIntercepts[2]*r3 + (1.0-RConicYIntercepts[2])*r1
        if isGreater(r, 0.0)
            RPArray[3] += 1
        elseif isClose(r, 0.0)
            RPArray[3] += 1            
            RNArray[3] += 1
        else
            RNArray[3] += 1
        end

    end

    if 1e-10 < RConicHIntercepts[1] < 1.0-1e-10 && !isClose(dot(normalizedGradient(RConic, RConicHIntercepts[1], 1.0-RConicHIntercepts[1]), (-1.0/sqrt(2),1.0/sqrt(2))), 0.0)

        r = RConicHIntercepts[1]*r2 + (1.0-RConicHIntercepts[1])*r3
        if isGreater(r, 0.0)
            RPArray[2] += 1
        elseif isClose(r, 0.0)
            RPArray[2] += 1            
            RNArray[2] += 1
        else
            RNArray[2] += 1
        end
    end

    if 1e-10 < RConicHIntercepts[2] < 1.0-1e-10 && !isClose(dot(normalizedGradient(RConic, RConicHIntercepts[2], 1.0-RConicHIntercepts[2]), (-1.0/sqrt(2),1.0/sqrt(2))), 0.0) &&
        RConicHIntercepts[1] != RConicHIntercepts[2]

        r = RConicHIntercepts[2]*r2 + (1.0-RConicHIntercepts[2])*r3
        if isGreater(r, 0.0)
            RPArray[2] += 1
        elseif isClose(r, 0.0)
            RPArray[2] += 1            
            RNArray[2] += 1
        else
            RNArray[2] += 1
        end
    end

    class, center = classifyAndReturnCenter(RConic)

    if class == ELLIPSE && RPArray == MArray{Tuple{3}, Int8}(zeros(Int8, 3)) && RNArray == MArray{Tuple{3}, Int8}(zeros(Int8, 3))
        if is_inside_triangle(center[1], center[2])
            if (r2-r1)*center[1] + (r3-r1)*center[2] + r1 > 0
                RPArray[1] = INTERNAL_ELLIPSE
            else
                RNArray[1] = INTERNAL_ELLIPSE
            end
        end
    end

    return cellTopologyEigenvector(vertexTypes, RPArray, RNArray)
end

end