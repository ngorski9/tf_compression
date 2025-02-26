module cellTopology

using ..utils
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
    DPArray::SArray{Tuple{10}, Int8}
    DNArray::SArray{Tuple{10}, Int8}
    RPArray::SArray{Tuple{10}, Int8}
    RNArray::SArray{Tuple{10}, Int8}
    RPArrayVec::SArray{Tuple{3}, Int8} # stores number of intersections with each edge
    RNArrayVec::SArray{Tuple{3}, Int8}
    hits_corners::SArray{Tuple{3}, Bool}
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
const INTERNAL_ELLIPSE::Int8 = 8
const STRAIGHT_ANGLES::Int8 = 9
# unfortunate intersection codes that I added by necessity :((
const CORNER_13::Int8 = 10 # when one conic intersects a corner
const CORNER_12::Int8 = 11
const CORNER_23::Int8 = 12
const CORNER_13_Z::Int8 = 13 # when a conic intersects a corner and is zero.
const CORNER_12_Z::Int8 = 14
const CORNER_23_Z::Int8 = 15
const E1Z::Int8 = 16 # when the entire tensor is equal to 0 at some point alone an edge.
const E2Z::Int8 = 17
const E3Z::Int8 = 18


# for eigenvalue, corners do not count as intersections, because
# the intersection of a region with a corner is given by the vertex classification.

# used for specifying that certain intersections are invalid / do not count.
const NULL::Int8 = -1

# vertex types (eigenvalue) (not all are actually used for classifying the corners, but all are used in a related context.)
const DP::Int8 = 19
const DN::Int8 = 20
const RP::Int8 = 21
const RN::Int8 = 22
const S::Int8 = 23
const RPTrumped::Int8 = 24 # used for detecting P vs N for vertex eigenvector
const RZTrumped::Int8 = 25
const RNTrumped::Int8 = 26
const DZ::Int8 = 27 # used for detecting degenerate intersections.
const RZ::Int8 = 28
const DREQP::Int8 = 29 # D and R are equal, the one that we are looking at is positive (based on context) Note: only used for edge and corner intersections.
const DREQN::Int8 = 30
const DREQZ::Int8 = 31

# vertex types (eigenvector)
const RRP::Int8 = 32
const DegenRP::Int8 = 33
const SRP::Int8 = 34
const SYM::Int8 = 35
const SRN::Int8 = 36
const DegenRN::Int8 = 37
const RRN::Int8 = 38
const Z::Int8 = 39

# vertex types (eigenvalue) only used for checking individual vertices and not cells.
const DPRPS::Int8 = 40 # all 3 are equal, D and R both positive
const DPRNS::Int8 = 41
const DNRPS::Int8 = 42
const DNRNS::Int8 = 43
const DPS::Int8 = 44
const DNS::Int8 = 45
const RPS::Int8 = 46
const RNS::Int8 = 47

export BLANK
export E1
export E2
export E3
export DPRP
export DPRN
export DNRP
export DNRN
export INTERNAL_ELLIPSE
export STRAIGHT_ANGLES
export CORNER_13
export CORNER_12
export CORNER_23
export CORNER_13_Z
export CORNER_12_Z
export CORNER_23_Z
export E1Z
export E2Z
export E3Z
export NULL
export DP
export DN
export RP
export RN
export S
export RRP
export DegenRP
export SRP
export SYM
export SRN
export DegenRN
export RRN
export Z
export DPRPS
export DPRNS
export DNRPS
export DNRNS
export DPS
export DNS
export RPS
export RNS

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

# more gradient based correction may occur later on if two are equal.
# This breaks ties arbitrarily.
function classifyTensorEigenvalue(d,r,s)
    if isGreater(abs(d), abs(r))
        if !isLess(s, abs(d))
            return S
        else
            if isGreater(d, 0.0)
                return DP
            else
                return DN
            end
        end
    else
        if !isLess(s, abs(r))
            if isClose(s, 0.0)
                return Z
            else
                return S
            end
        else
            if isGreater(r, 0.0)
                return RP
            else
                return RN
            end
        end
    end
end

function classifyTensorEigenvector(r,s)
    if isClose(r, 0.0)
        if isClose(s, 0.0)
            return Z
        else
            return SYM
        end
    elseif isGreater(r,0.0)
        if isClose(r,s)
            return DegenRP
        elseif isGreater(r,s)
            return RRP
        else
            return SRP
        end
    else
        if isClose(-r,s)
            return DegenRN
        elseif isGreater(-r,s)
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
    if isClose(abs(r),abs(d))
        if isGreater(d,0.0)
            return DREQP
        elseif isClose(d,0.0)
            return DZ
        else
            return DREQN
        end
    elseif isGreater(abs(r),abs(d))
        return NULL
    elseif isGreater(d, 0.0)
        return DP
    elseif isClose(d, 0.0)
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

    if isClose(abs(d),abs(r))
        if isGreater(r,0.0)
            return DREQP
        elseif isClose(r,0.0)
            return DZ
        else
            return DREQN
        end
    elseif isGreater(abs(d),abs(r))
        if isGreater(r,0.0)
            return RPTrumped
        elseif isClose(r,0.0)
            return RZTrumped
        else
            return RNTrumped
        end
    else
        if isGreater(r,0.0)
            return RP
        elseif isClose(r,0.0)
            return RZ
        else
            return RN
        end
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
macro pushCodeFromSign(PList, NList, PListSize, NListSize, point, crossingCode, signCode, positiveTest, negativeTest, entering, tangent_vector, alt_conic, edge_vector)
    return :(
        if $(esc(signCode)) == $positiveTest
            $(esc(PListSize)) += 1
            $(esc(PList))[$(esc(PListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode)
        elseif $(esc(signCode)) == $negativeTest
            $(esc(NListSize)) += 1
            $(esc(NList))[$(esc(NListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode)
        elseif $(esc(signCode)) != NULL
            alt_grad = normalizedGradient($(esc(alt_conic)), $(esc(point))[1], $(esc(point))[2])
            dot_ = dot(alt_grad, $(esc(edge_vector)))
            if !isClose(abs(dot_), 1.0) || (dot_ < 0.0)
                if $(esc(signCode)) == DREQP
                    $(esc(PListSize)) += 1
                    $(esc(PList))[$(esc(PListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode)
                elseif $(esc(signCode)) == DREQN
                    $(esc(NListSize)) += 1
                    $(esc(NList))[$(esc(NListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode)
                end
            end
        end
    )
end

macro pushCodeFromSignZero(PList, NList, PListSize, NListSize, point, crossingCode, crossingCodeZero, signCode, positiveTest, negativeTest, zeroTest, entering, tangent_vector, alt_conic, edge_vector1, edge_vector2, hits_corners, corner_index)
    return :(
        if $(esc(signCode)) == $positiveTest
            $(esc(PListSize)) += 1
            $(esc(PList))[$(esc(PListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode)
            $(esc(hits_corners))[$corner_index] = true
        elseif $(esc(signCode)) == $negativeTest
            $(esc(NListSize)) += 1
            $(esc(NList))[$(esc(NListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode)
            $(esc(hits_corners))[$corner_index] = true
        elseif $(esc(signCode)) == $zeroTest
            $(esc(PListSize)) += 1
            $(esc(NListSize)) += 1            
            $(esc(PList))[$(esc(PListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], $crossingCodeZero)
            $(esc(NList))[$(esc(NListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], $crossingCodeZero)
            $(esc(hits_corners))[$corner_index] = true
        elseif $(esc(signCode)) != NULL
            alt_grad = normalizedGradient($(esc(alt_conic)), $(esc(point))[1], $(esc(point))[2])
            dot1 = dot(alt_grad, $(esc(edge_vector1)))
            dot2 = dot(alt_grad, $(esc(edge_vector2)))

            if !(isClose(abs(dot1), 1.0) || isClose(abs(dot2), 1.0)) || (dot1 < 0.0 || dot2 < 0.0)
                if $(esc(signCode)) == DREQP
                    $(esc(PListSize)) += 1
                    $(esc(PList))[$(esc(PListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode)
                    $(esc(hits_corners))[$corner_index] = true
                elseif $(esc(signCode)) == DREQN
                    $(esc(NListSize)) += 1
                    $(esc(NList))[$(esc(NListSize))] = Intersection($(esc(point))[1], $(esc(point))[2], Int8($(esc(entering))) * $crossingCode)
                    $(esc(hits_corners))[$corner_index] = true
                end
            end
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
                return !d
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
    return :(begin
        if $(esc(code)) == RP || $(esc(code)) == RPTrumped || $(esc(code)) == DREQP
            $(esc(eigenvectorP))[$E] += 1
        elseif $(esc(code)) == RZ || $(esc(code)) == RZTrumped
            $(esc(eigenvectorP))[$E] += 1
            $(esc(eigenvectorN))[$E] += 1
        elseif $(esc(code)) == RN || $(esc(code)) == RNTrumped || $(esc(code)) == DREQN
            $(esc(eigenvectorN))[$E] += 1
        end
    end)
end

# yes this is absolutely horrendus but the alternative is to write this out six times which is even worse.
# writing one macro that works is far less glitch prone
macro process_intercepts(edge_number, is_d, intercepts, alt_list, class_fun, d1, d2, r1, r2, PIntercepts, NIntercepts, PInterceptsSize, NInterceptsSize, conic, alt_conic, check_low, check_high,
                         do_eigenvector, do_eigenvector_runtime, eigenvectorP, eigenvectorN, any_intercepts, ignore_other, hits_corners)

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
        low_index = 1
        high_index = 2
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
        low_index = 3
        high_index = 2
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
        low_index = 1
        high_index = 3
    end

    return :(begin
    if -ϵ <= $(esc(intercepts))[1] <= 1.0 + ϵ
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

                @pushCodeFromSignZero($(esc(PIntercepts)), $(esc(NIntercepts)), $(esc(PInterceptsSize)), $(esc(NInterceptsSize)), $low_coords, $CORNER_L, $CORNER_L_Z, class, $P, $N, $Z, sign(entering), tangentVector, $(esc(alt_conic)), $edge_inside, $low_edge_inside, $(esc(hits_corners)), $low_index)

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

                @pushCodeFromSignZero($(esc(PIntercepts)), $(esc(NIntercepts)), $(esc(PInterceptsSize)), $(esc(NInterceptsSize)), $high_coords, $CORNER_H, $CORNER_H_Z, class, $P, $N, $Z, sign(entering), tangentVector, $(esc(alt_conic)), $edge_inside, $high_edge_inside, $(esc(hits_corners)), $high_index)
            end

        elseif isClose(dot(tangentVector,$edge_inside ),0.0) || isnan(tangentVector[1]) # e.g. if we have a non-transverse intersection
            if class == $Z
                $(esc(PInterceptsSize)) += 1
                $(esc(NInterceptsSize)) += 1
                $(esc(PIntercepts))[$(esc(PInterceptsSize))] = Intersection($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1]), $EZ)
                $(esc(NIntercepts))[$(esc(NInterceptsSize))] = Intersection($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1]), $EZ)

                if $do_eigenvector && $(esc(do_eigenvector_runtime))
                    $(esc(eigenvectorP))[$E] += 1
                    $(esc(eigenvectorN))[$E] += 1
                end
            end
        else
            if (!isClose($(esc(alt_list))[1],$(esc(intercepts))[1]) && !isClose($(esc(alt_list))[2],$(esc(intercepts))[1])) || isClose($(esc(alt_list))[1],$(esc(alt_list))[2]) || 
                doesConicEquationCrossDoubleBoundary($(esc(conic)), $(esc(alt_conic)), ($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1])), $edge_orientation, $edge_inside, tangentVector, $is_d)

                entering = dot(tangentVector, $edge_inside)

                @pushCodeFromSign($(esc(PIntercepts)), $(esc(NIntercepts)), $(esc(PInterceptsSize)), $(esc(NInterceptsSize)), ($x($(esc(intercepts))[1]), $y($(esc(intercepts))[1])), $E, class, $P, $N, sign(entering), tangentVector, $(esc(alt_conic)), $edge_inside)

            end
            if $do_eigenvector && $(esc(do_eigenvector_runtime))

                @eigenvector_push($(esc(eigenvectorP)), $(esc(eigenvectorN)), $E, class)

            end
        end
    end

    if -ϵ <= $(esc(intercepts))[2] <= 1.0 + ϵ && $(esc(intercepts))[1] != $(esc(intercepts))[2]
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

                @pushCodeFromSignZero($(esc(PIntercepts)), $(esc(NIntercepts)), $(esc(PInterceptsSize)), $(esc(NInterceptsSize)), $low_coords, $CORNER_L, $CORNER_L_Z, class, $P, $N, $Z, sign(entering), tangentVector, $(esc(alt_conic)), $edge_inside, $low_edge_inside, $(esc(hits_corners)), $low_index)

            end

        elseif isClose($(esc(intercepts))[2],1.0)
            if $check_high && (class == $Z || (
            doesConicEquationCrossCorner($(esc(conic)), $(high_coords)[1], $(high_coords)[2], $edge_inside, $high_edge_inside, grad, tangentVector) && 
            ($(esc(ignore_other)) || (!isClose($(esc(alt_list))[1],$(esc(intercepts))[2]) && !isClose($(esc(alt_list))[2],$(esc(intercepts))[2])) || 
                doesConicEquationCrossDoubleBoundary($(esc(conic)), $(esc(alt_conic)), ($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2])), (-$edge_orientation[1], -$edge_orientation[2]), $edge_inside, tangentVector, $is_d)
            )))

                entering = dot(tangentVector, $edge_inside)
                if isClose(entering, 0.0)
                    entering = dot(tangentVector, $high_edge_inside)
                end

                @pushCodeFromSignZero($(esc(PIntercepts)), $(esc(NIntercepts)), $(esc(PInterceptsSize)), $(esc(NInterceptsSize)), $high_coords, $CORNER_H, $CORNER_H_Z, class, $P, $N, $Z, sign(entering), tangentVector, $(esc(alt_conic)), $edge_inside, $high_edge_inside, $(esc(hits_corners)), $high_index)

            end
        elseif isClose(dot(tangentVector,$edge_inside ),0.0)  || isnan(tangentVector[1]) # e.g. if we have a non-transverse intersection
            if class == $Z         
                $(esc(PInterceptsSize)) += 1
                $(esc(NInterceptsSize)) += 1
                $(esc(PIntercepts))[$(esc(PInterceptsSize))] = Intersection($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2]), $EZ)
                $(esc(NIntercepts))[$(esc(NInterceptsSize))] = Intersection($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2]), $EZ)

                if $do_eigenvector && $(esc(do_eigenvector_runtime))
                    $(esc(eigenvectorP))[$E] += 1
                    $(esc(eigenvectorN))[$E] += 1
                end
            end
        else   
            if (!isClose($(esc(alt_list))[1],$(esc(intercepts))[2]) && !isClose($(esc(alt_list))[2],$(esc(intercepts))[2])) || isClose($(esc(alt_list))[1],$(esc(alt_list))[2]) || 
                doesConicEquationCrossDoubleBoundary($(esc(conic)), $(esc(alt_conic)), ($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2])), $edge_orientation, $edge_inside, tangentVector, $is_d)
                entering = dot(tangentVector, $edge_inside)

                @pushCodeFromSign($(esc(PIntercepts)), $(esc(NIntercepts)), $(esc(PInterceptsSize)), $(esc(NInterceptsSize)), ($x($(esc(intercepts))[2]), $y($(esc(intercepts))[2])), $E, class, $P, $N, sign(entering), tangentVector, $(esc(alt_conic)), $edge_inside)
            end
            if $do_eigenvector && $(esc(do_eigenvector_runtime))
                @eigenvector_push($(esc(eigenvectorP)), $(esc(eigenvectorN)), $E, class)
            end
        end
    end
    end)
end

# Checks for and accounts for cases where d, r, or s are equal at a corner.
macro checkEqualityAtCorner(corner, d1, d2, d3, r1, r2, r3, sBase, DConic, RConic, class_list, ignore_d, ignore_r)
    s = esc(sBase)

    if corner == 1
        x = 0.0
        y = 0.0
        edge_1 = (1.0,0.0)
        edge_2 = (0.0,1.0)        
        d = esc(d1)
        r = esc(r1)
    elseif corner == 2
        x = 1.0
        y = 0.0
        edge_1 = (-1.0,1.0)
        edge_2 = (-1.0,0.0)
        d = esc(d2)
        r = esc(r2)
    else
        x = 0.0
        y = 1.0
        edge_1 = (0.0,-1.0)
        edge_2 = (1.0,-1.0)
        d = esc(d3)
        r = esc(r3)
    end

    return :(begin
        do_d = false
        do_r = false
        do_s = false

        if !$(esc(ignore_d)) && !$(esc(ignore_r)) && isClose(abs($r),abs($d)) && isGreater(abs($r),$s)
            do_d = true
            do_r = true
        else
            if !$(esc(ignore_d)) && isClose(abs($d), $s) && !isGreater(abs($r), abs($d))
                d_grad = normalizedGradient($(esc(DConic)), $x, $y)

                if isGreater(dot(d_grad, $edge_1), 0.0) || isGreater(dot(d_grad, $edge_2), 0.0)
                    do_d = true
                    do_s = true
                end
            end

            if !$(esc(ignore_r)) && isClose(abs($r), $s) && !isGreater(abs($d), abs($r))
                r_grad = normalizedGradient($(esc(RConic)), $x, $y)

                if isGreater(dot(r_grad, $edge_1), 0.0) || isGreater(dot(r_grad, $edge_2), 0.0)
                    do_r = true
                    do_s = true
                end
            end
        end

        if do_r
            if do_d

                if isClose($r,0.0)
                    $(esc(class_list))[$(esc(corner))] = Z
                else

                    # We use another conic here to represent the region where |d| > |r|, which we know is a double line
                    # I guess it would be more efficient to calc it by hand (in terms of machine performance), but honestly
                    # it will make the code a lot cleaner.

                    region_type = 0 # 0 = d dominates, 1 = r_dominates, 2 = mixed

                    DBase = interpolationConic($(esc(d1)), $(esc(d2)), $(esc(d3)))
                    RBase = interpolationConic($(esc(r1)), $(esc(r2)), $(esc(r3)))
                    RDConic = sub(DBase, RBase)
                    RDGrad = normalizedGradient(RDConic, $x, $y) # points in twoards d

                    dot1 = dot(RDGrad, $edge_1)
                    dot2 = dot(RDGrad, $edge_2)

                    if !isLess(dot1, 0.0) && !isLess(dot2, 0.0)
                        # in this case, the entire region is in the RD region
                        region_type = 0
                    elseif !isGreater(dot1, 0.0) && !isGreater(dot2, 0.0)
                        # in this case, the entire region is outside the RD region
                        region_type = 1
                    else
                        # in this case, the RD region intersects only part of the cell. However, all region types are still possible.

                        if !do_s
                            region_type = 2
                        else

                            DGrad = normalizedGradient($(esc(DConic)), $x, $y)
                            DTangent = tangentDerivative(DGrad)

                            if isLess(dot(DTangent,$edge_1), 0.0) || isLess(dot(DTangent,$edge_2), 0.0)
                                DTangent = (-DTangent[1], -DTangent[2])
                            end

                            RGrad = normalizedGradient($(esc(RConic)), $x, $y)
                            RTangent = tangentDerivative(RGrad)

                            if isLess(dot(RTangent,$edge_1), 0.0) || isLess(dot(RTangent,$edge_2), 0.0)
                                RTangent = (-RTangent[1], -RTangent[2])
                            end

                            # first, check whether the RD region and d region have gradients pointing in opposite directions
                            if isLess(dot(RDGrad, DGrad), 0.0)
                                if isGreater(dot(RDGrad, DTangent), 0.0) # point in opposite directions, but overlap
                                    region_type = 2
                                else # opposite directions & no overlap
                                    region_type = 1
                                end
                            else
                                # if the RD region and d region have gradients pointing in the same direction, then
                                # we are guaranteed to have at least some d. The d may envelop the r, or it may be mixed.
                                if isLess(dot(RGrad, RDGrad), 0.0)
                                    # r and d regions point in opposite directions, so it must be mixed
                                    region_type = 2
                                else
                                    if isGreater(dot(RTangent, DGrad), 0.0) && isGreater(dot(RTangent, RDGrad), 0.0)
                                        # in this case, the r region is contained entirely inside the d region
                                        region_type = 0
                                    else
                                        region_type = 2
                                    end
                                end
                            end
                        end

                    end

                    if region_type == 0
                        if $d > 0.0
                            $(esc(class_list))[$(esc(corner))] = DP
                        else
                            $(esc(class_list))[$(esc(corner))] = DN
                        end
                    elseif region_type == 1
                        if $r > 0.0
                            $(esc(class_list))[$(esc(corner))] = RP
                        else
                            $(esc(class_list))[$(esc(corner))] = RN
                        end
                    else
                        if $d > 0.0
                            if $r > 0.0
                                $(esc(class_list))[$(esc(corner))] = DPRP
                            else
                                $(esc(class_list))[$(esc(corner))] = DPRN
                            end
                        else
                            if $r > 0.0
                                $(esc(class_list))[$(esc(corner))] = DNRP
                            else
                                $(esc(class_list))[$(esc(corner))] = DNRN
                            end
                        end
                    end

                end


            else
                
                # just do r and not d

                if isGreater($r, 0.0)
                    $(esc(class_list))[$(esc(corner))] = RP
                elseif isLess($r, 0.0)
                    $(esc(class_list))[$(esc(corner))] = RN
                else
                    $(esc(class_list))[$(esc(corner))] = Z
                end

            end
        elseif do_d

            if isGreater($d, 0.0)
                $(esc(class_list))[$(esc(corner))] = DP
            elseif isLess($d, 0.0)
                $(esc(class_list))[$(esc(corner))] = DN
            else
                $(esc(class_list))[$(esc(corner))] = Z
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

# While technically we use a barycentric interpolation scheme which is agnostic to the locations of the actual cell vertices,
# for mathematical ease we assume that point 1 is at (0,0), point 2 is at (1,0), and point 3 is at (0,1). Choosing a specific
# embedding will not affect the topology.
# The final bool tells whether we simultaneously compute eigenvector topology.
function classifyCellEigenvalue(M1::SMatrix{2,2,Float64}, M2::SMatrix{2,2,Float64}, M3::SMatrix{2,2,Float64},eigenvector::Bool,verbose::Bool=false)
    DPArray = MArray{Tuple{10}, Int8}((0,0,0,0,0,0,0,0,0,0))
    DNArray = MArray{Tuple{10}, Int8}((0,0,0,0,0,0,0,0,0,0))
    RPArray = MArray{Tuple{10}, Int8}((0,0,0,0,0,0,0,0,0,0))
    RNArray = MArray{Tuple{10}, Int8}((0,0,0,0,0,0,0,0,0,0))
    RPArrayVec = MArray{Tuple{3}, Int8}((0,0,0))
    RNArrayVec = MArray{Tuple{3}, Int8}((0,0,0))
    hits_corners = MArray{Tuple{3}, Bool}((false,false,false))

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

    vertexTypesEigenvalue = MArray{Tuple{3},Int8}((classifyTensorEigenvalue(d1,r1,s1), classifyTensorEigenvalue(d2,r2,s2), classifyTensorEigenvalue(d3,r3,s3)))

    if eigenvector
        vertexTypesEigenvector = SArray{Tuple{3},Int8}((classifyTensorEigenvector(r1,s1), classifyTensorEigenvector(r2,s2), classifyTensorEigenvector(r3, s3)))
    else
        vertexTypesEigenvector = SArray{Tuple{3},Int8}((0,0,0))
    end

    if (vertexTypesEigenvalue[1] == Z && vertexTypesEigenvalue[2] == Z && vertexTypesEigenvalue[3] == Z) ||
       ( (( !eigenvector && isGreater(abs(d1),s1) && isGreater(abs(d2),s2) && isGreater(abs(d3),s3) && ( ( isGreater(d1,0.0) && isGreater(d2,0.0) && isGreater(d3,0.0)) || ( isLess(d1,0.0) && isLess(d2,0.0) && isLess(d3,0.0) ) ) ) ||
            ( isGreater(abs(r1),s1) && isGreater(abs(r2),s2) && isGreater(abs(r3),s3) && ( ( isGreater(r1,0.0) && isGreater(r2,0.0) && isGreater(r3,0.0) ) || ( isLess(r1,0.0) && isLess(r2,0.0) && isLess(r3,0.0) ) ) )) &&
            (!isClose(abs(d1),abs(r1)) && !isClose(abs(d2),abs(r2)) && !isClose(abs(d3),abs(r3))) )   ||
       (isClose(s1,0.0) && isClose(s2,0.0) && isClose(s3,0.0))
       # in this case, s is dominated by d or r throughout the entire triangle, so the topology follows from the vertices.
    #    println("return 1")
        return cellTopologyEigenvalue(vertexTypesEigenvalue, vertexTypesEigenvector, DPArray, DNArray, RPArray, RNArray, RPArrayVec, RNArrayVec, hits_corners)
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

    ignore_d = (d_type == POINT || d_type == EMPTY || d_type == LINE_NO_REGION || (isClose(r1,d1) && isClose(r2,d2) && isClose(r3,d3)) || (isClose(r1,-d1) && isClose(r2,-d2) && isClose(r3,-d3)))
    ignore_r = (r_type == POINT || r_type == EMPTY || r_type == LINE_NO_REGION)

    # third elt is true for double root, false otherwise
    DPIntercepts = MArray{Tuple{6},Intersection}(undef)
    DNIntercepts = MArray{Tuple{6},Intersection}(undef)
    RPIntercepts = MArray{Tuple{6},Intersection}(undef)
    RNIntercepts = MArray{Tuple{6},Intersection}(undef)
    DPInterceptsSize = 0
    DNInterceptsSize = 0
    RPInterceptsSize = 0
    RNInterceptsSize = 0

    # at this point the loop unrolling was so bad that we had to outsource to macros. It is truly awful.

    # process D intercepts

    any_d_intercepts = false
    any_r_intercepts = false

    if !ignore_d
        # process D intercepts
        # (edge_number, is_d, intercepts, alt_list, class_fun, d1, d2, r1, r2, PIntercepts, NIntercepts, conic, alt_conic, check_low, check_high,
        # do_eigenvector, do_eigenvector_runtime, eigenvector_c, eigenvectorP, eigenvectorN, any_intercepts, ignore_other)

        @process_intercepts(1, true, DConicXIntercepts, RConicXIntercepts, DCellIntersection, d2, d1, r2, r1, DPIntercepts, DNIntercepts, DPInterceptsSize, DNInterceptsSize, DConic, RConic, true, true, 
        false, false, RPArrayVec, RNArrayVec, any_d_intercepts, ignore_r, hits_corners)

        @process_intercepts(2, true, DConicHIntercepts, RConicHIntercepts, DCellIntersection, d2, d3, r2, r3, DPIntercepts, DNIntercepts, DPInterceptsSize, DNInterceptsSize, DConic, RConic, false, false, 
        false, false, RPArrayVec, RNArrayVec, any_d_intercepts, ignore_r, hits_corners)

        @process_intercepts(3, true, DConicYIntercepts, RConicYIntercepts, DCellIntersection, d3, d1, r3, r1, DPIntercepts, DNIntercepts, DPInterceptsSize, DNInterceptsSize, DConic, RConic, false, true, 
        false, false, RPArrayVec, RNArrayVec, any_d_intercepts, ignore_r, hits_corners)
    end

    if !ignore_r
        # process R intercepts
        @process_intercepts(1, false, RConicXIntercepts, DConicXIntercepts, RCellIntersection, d2, d1, r2, r1, RPIntercepts, RNIntercepts, RPInterceptsSize, RNInterceptsSize, RConic, DConic, true, true, 
        true, eigenvector, RPArrayVec, RNArrayVec, any_r_intercepts, ignore_d, hits_corners)

        @process_intercepts(2, false, RConicHIntercepts, DConicHIntercepts, RCellIntersection, d2, d3, r2, r3, RPIntercepts, RNIntercepts, RPInterceptsSize, RNInterceptsSize, RConic, DConic, false, false, 
        true, eigenvector, RPArrayVec, RNArrayVec, any_r_intercepts, ignore_d, hits_corners)

        @process_intercepts(3, false, RConicYIntercepts, DConicYIntercepts, RCellIntersection, d3, d1, r3, r1, RPIntercepts, RNIntercepts, RPInterceptsSize, RNInterceptsSize, RConic, DConic, false, true, 
        true, eigenvector, RPArrayVec, RNArrayVec, any_r_intercepts, ignore_d, hits_corners)
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

    if !any_d_intercepts && !any_r_intercepts && !d_internal_ellipse && !r_internal_ellipse && vertexTypesEigenvalue[1] == S &&
        vertexTypesEigenvector[1] != DegenRP && vertexTypesEigenvector[2] != DegenRP && vertexTypesEigenvector[3] != DegenRP &&
        vertexTypesEigenvector[1] != DegenRN && vertexTypesEigenvector[2] != DegenRN && vertexTypesEigenvector[3] != DegenRN
        # println("return 2")
        return cellTopologyEigenvalue(SArray{Tuple{3},Int8}(S,S,S), vertexTypesEigenvector, 
            SArray{Tuple{10},Int8}(0,0,0,0,0,0,0,0,0,0), SArray{Tuple{10},Int8}(0,0,0,0,0,0,0,0,0,0), SArray{Tuple{10},Int8}(0,0,0,0,0,0,0,0,0,0), 
            SArray{Tuple{10},Int8}(0,0,0,0,0,0,0,0,0,0), SArray{Tuple{3},Int8}(0,0,0), SArray{Tuple{3},Int8}(0,0,0), SArray{Tuple{3},Bool}(false,false,false))
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
        for i in 1:DPInterceptsSize
            push!(DPPoints, ellipse_intersection((DPIntercepts[i].x, DPIntercepts[i].y), d_center, DVector1, DVector2, DPIntercepts[i].code))
        end

        for i in 1:DNInterceptsSize
            push!(DNPoints, ellipse_intersection((DNIntercepts[i].x, DNIntercepts[i].y), d_center, DVector1, DVector2, DNIntercepts[i].code))
        end
    else
        # verbose, but I need to reduce the number of checks that happen
        for i in 1:DPInterceptsSize
            @orientHyperbolaAndPush((DPIntercepts[i].x,DPIntercepts[i].y),d_center,DVector1,DVector2,d_orientation,DPPoints,DPIntercepts[i].code,true)
        end

        for i in 1:DNInterceptsSize
            @orientHyperbolaAndPush((DNIntercepts[i].x,DNIntercepts[i].y),d_center,DVector1,DVector2,d_orientation,DNPoints,DNIntercepts[i].code,false)
        end
    end

    if r_ellipse
        for i in 1:RPInterceptsSize
            push!(RPPoints, ellipse_intersection((RPIntercepts[i].x, RPIntercepts[i].y), r_center, RVector1, RVector2, RPIntercepts[i].code))
        end

        for i in 1:RNInterceptsSize
            push!(RNPoints, ellipse_intersection((RNIntercepts[i].x, RNIntercepts[i].y), r_center, RVector1, RVector2, RNIntercepts[i].code))
        end
    else
        # verbose, but I need to reduce the number of checks that happen
        for i in 1:RPInterceptsSize
            @orientHyperbolaAndPush((RPIntercepts[i].x,RPIntercepts[i].y),r_center,RVector1,RVector2,r_orientation,RPPoints,RPIntercepts[i].code,true)
        end

        for i in 1:RNInterceptsSize
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

    if !any_r_intercepts && r_type == ELLIPSE && is_inside_triangle(r_center[1], r_center[2]) && abs(r1) < s1
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

    # fix the corners as needed
    @checkEqualityAtCorner(1, d1, d2, d3, r1, r2, r3, s1, DConic, RConic, vertexTypesEigenvalue, ignore_d, ignore_r)
    @checkEqualityAtCorner(2, d1, d2, d3, r1, r2, r3, s2, DConic, RConic, vertexTypesEigenvalue, ignore_d, ignore_r)
    @checkEqualityAtCorner(3, d1, d2, d3, r1, r2, r3, s3, DConic, RConic, vertexTypesEigenvalue, ignore_d, ignore_r)

    # what a racket
    # println("return 3")
    return cellTopologyEigenvalue(vertexTypesEigenvalue, vertexTypesEigenvector, DPArray, DNArray, RPArray, RNArray, RPArrayVec, RNArrayVec, hits_corners)
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

    # check if the conic is not basically null (I don't believe this is even possible.)
    class, _ = classifyAndReturnCenter(RConic)
    if class == POINT || class == EMPTY || class == LINE_NO_REGION
        return cellTopologyEigenvector(vertexTypes, RPArray, RNArray)
    end

    RConicXIntercepts = quadraticFormula(RConic.A, RConic.D, RConic.F) # gives x coordinate
    RConicYIntercepts = quadraticFormula(RConic.C, RConic.E, RConic.F) # gives y coordinate
    # hypotenuse intercepts. Gives x coordinate
    RConicHIntercepts = quadraticFormula(RConic.A - RConic.B + RConic.C, RConic.B - 2*RConic.C + RConic.D - RConic.E, RConic.C + RConic.E + RConic.F)

    if ϵ < RConicXIntercepts[1] < 1.0-ϵ
        r = RConicXIntercepts[1]*r2 + (1.0-RConicXIntercepts[1])*r1

        grad = normalizedGradient(RConic, RConicXIntercepts[1], 0.0)

        if isClose(dot(grad, (1.0,0.0)), 0.0) || isnan(grad[1])
            if isClose(r, 0.0)
                RPArray[1] += 1
                RNArray[1] += 1
            end
        else
            if isGreater(r, 0.0)
                RPArray[1] += 1
            elseif isClose(r, 0.0)
                RPArray[1] += 1            
                RNArray[1] += 1
            else
                RNArray[1] += 1
            end
        end
    end

    if ϵ < RConicXIntercepts[2] < 1.0-ϵ && RConicXIntercepts[1] != RConicXIntercepts[2]
        

        r = RConicXIntercepts[2]*r2 + (1.0-RConicXIntercepts[2])*r1
        
        grad = normalizedGradient(RConic, RConicXIntercepts[2], 0.0)

        if isClose(dot(grad, (1.0,0.0)), 0.0) || isnan(grad[1])
            if isClose(r, 0.0)
                RPArray[1] += 1
                RNArray[1] += 1
            end
        else
            if isGreater(r, 0.0)
                RPArray[1] += 1
            elseif isClose(r, 0.0)
                RPArray[1] += 1            
                RNArray[1] += 1
            else
                RNArray[1] += 1
            end
        end
    end

    if ϵ < RConicYIntercepts[1] < 1.0-ϵ

        r = RConicYIntercepts[1]*r3 + (1.0-RConicYIntercepts[1])*r1

        grad = normalizedGradient(RConic, 0.0, RConicYIntercepts[1])

        if isClose(dot(grad, (0.0,1.0)), 0.0) || isnan(grad[1])
            if isClose(r, 0.0)
                RPArray[1] += 1
                RNArray[1] += 1
            end
        else
            if isGreater(r, 0.0)
                RPArray[3] += 1
            elseif isClose(r, 0.0)
                RPArray[3] += 1            
                RNArray[3] += 1
            else
                RNArray[3] += 1
            end
        end
    end

    if ϵ < RConicYIntercepts[2] < 1.0-ϵ && RConicYIntercepts[1] != RConicYIntercepts[2]

        r = RConicYIntercepts[2]*r3 + (1.0-RConicYIntercepts[2])*r1

        grad = normalizedGradient(RConic, 0.0, RConicYIntercepts[2])

        if isClose(dot(grad, (0.0,1.0)), 0.0) || isnan(grad[1])
            if isClose(r, 0.0)
                RPArray[1] += 1
                RNArray[1] += 1
            end
        else
            if isGreater(r, 0.0)
                RPArray[3] += 1
            elseif isClose(r, 0.0)
                RPArray[3] += 1            
                RNArray[3] += 1
            else
                RNArray[3] += 1
            end
        end

    end

    if ϵ < RConicHIntercepts[1] < 1.0-ϵ

        r = RConicHIntercepts[1]*r2 + (1.0-RConicHIntercepts[1])*r3

        grad = normalizedGradient(RConic, RConicHIntercepts[1], 1.0-RConicHIntercepts[1])

        if isClose(dot(grad, (-1.0/sqrt(2),1.0/sqrt(2))), 0.0) || isnan(grad[1])
            if isClose(r, 0.0)
                RPArray[1] += 1
                RNArray[1] += 1
            end
        else
            if isGreater(r, 0.0)
                RPArray[2] += 1
            elseif isClose(r, 0.0)
                RPArray[2] += 1            
                RNArray[2] += 1
            else
                RNArray[2] += 1
            end
        end
    end

    if ϵ < RConicHIntercepts[2] < 1.0-ϵ && RConicHIntercepts[1] != RConicHIntercepts[2]

        r = RConicHIntercepts[2]*r2 + (1.0-RConicHIntercepts[2])*r3

        grad = normalizedGradient(RConic, RConicHIntercepts[2], 1.0-RConicHIntercepts[2])

        if isClose(dot(grad, (-1.0/sqrt(2),1.0/sqrt(2))), 0.0) || isnan(grad[1])
            if isClose(r, 0.0)
                RPArray[1] += 1
                RNArray[1] += 1
            end
        else
            if isGreater(r, 0.0)
                RPArray[2] += 1
            elseif isClose(r, 0.0)
                RPArray[2] += 1            
                RNArray[2] += 1
            else
                RNArray[2] += 1
            end
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

    if vertexTypes[1] == DegenRP || vertexTypes[2] == DegenRP || vertexTypes[3] == DegenRP
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

        if isClose(abs(sin(θ1)), abs(sin(θ2))) && vertexTypes[1] == DegenRP && vertexTypes[2] == DegenRP
            RPArray[1] = STRAIGHT_ANGLES
        end

        if isClose(abs(sin(θ2)), abs(sin(θ3))) && vertexTypes[2] == DegenRP && vertexTypes[3] == DegenRP
            RPArray[2] = STRAIGHT_ANGLES
        end

        if isClose(abs(sin(θ3)), abs(sin(θ1))) && vertexTypes[3] == DegenRP && vertexTypes[1] == DegenRP
            RPArray[3] = STRAIGHT_ANGLES
        end
    elseif vertexTypes[1] == DegenRN || vertexTypes[2] == DegenRN || vertexTypes[3] == DegenRN
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

        if isClose(abs(sin(θ1)), abs(sin(θ2))) && vertexTypes[1] == DegenRN && vertexTypes[2] == DegenRN
            RNArray[1] = STRAIGHT_ANGLES
        end

        if isClose(abs(sin(θ2)), abs(sin(θ3))) && vertexTypes[2] == DegenRN && vertexTypes[3] == DegenRN
            RNArray[2] = STRAIGHT_ANGLES
        end

        if isClose(abs(sin(θ3)), abs(sin(θ1))) && vertexTypes[3] == DegenRN && vertexTypes[1] == DegenRN
            RNArray[3] = STRAIGHT_ANGLES
        end            
    end

    return cellTopologyEigenvector(vertexTypes, RPArray, RNArray)
end

end