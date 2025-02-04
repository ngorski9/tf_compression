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
function Base.isless(first::Intersection, second::Intersection)
    if first.y >= 0
        if second.y >= 0
            return first.x > second.x
        else
            return true
        end
    else
        if second.y >= 0
            return false
        else
            return first.x < second.x
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
const RNTrumped = 6
const DZ::Int8 = 7 # used for detecting degenerate intersections.
const RZ::Int8 = 8

# vertex types (eigenvector)
const RRP::Int8 = 9
const DegenRP::Int8 = 10
const SRP::Int8 = 11
const SYM::Int8 = 12
const SRN::Int8 = 13
const DegenRN::Int8 = 14
const RRN::Int8 = 15
const Z::Int8 = 16 # all zeros.

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
    if abs(d) >= abs(r) && abs(d) >= s
        if d > 0
            return DP
        else
            return DN
        end
    elseif abs(r) >= abs(d) && abs(r) >= abs(s)
        if r > 0
            return RP
        else
            return RN
        end
    else
        return S
    end
end

function classifyTensorEigenvector(r,s)
    if r > 0.0
        if r > s
            return RRP
        elseif r == s
            return DegenRP
        else
            return SRP
        end
    elseif r == 0
        if s == 0.0
            return Z
        else
            return SYM
        end
    else
        if -r > s
            return RRN
        elseif -r == s
            return DegenRN
        else
            return SRN
        end
    end
end

function is_inside_triangle(x::Float64,y::Float64)
    return x >= 0.0 && y >= 0.0 && y <= 1.0-x
end

# returns the signs of d and r when the |d|=s and |r|=s curves intersect.
function DRSignAt(d1::Float64, d2::Float64, d3::Float64, x::Float64, y::Float64, same_sign::Bool)
    d = (d2-d1)*x + (d3-d1)*y + d1

    if same_sign
        if d > 0
            return DPRP
        else
            return DNRN
        end
    else
        if d > 0
            return DPRN
        else
            return DNRP
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
macro placeIntersectionPointInLists(point,d_center,r_center,DVector1,DVector2,RVector1,RVector2,DList,RList,d_ellipse,r_ellipse,d_orientation,r_orientation,code,d_positive,r_positive)
    return :(begin

    if $(esc(d_ellipse))
        push!($(esc(DList)), ellipse_intersection($(esc(point)), $(esc(d_center)), $(esc(DVector1)), $(esc(DVector2)), $(esc(code))))
    else
        @orientHyperbolaAndPush($(esc(point)),$(esc(d_center)),$(esc(DVector1)),$(esc(DVector2)),$(esc(d_orientation)),$(esc(DList)),$(esc(code)),$d_positive)
    end

    if $(esc(r_ellipse))
        push!($(esc(RList)), ellipse_intersection($(esc(point)), $(esc(r_center)), $(esc(RVector1)), $(esc(RVector2)), $(esc(code))))
    else
        @orientHyperbolaAndPush($(esc(point)),$(esc(r_center)),$(esc(RVector1)),$(esc(RVector2)),$(esc(r_orientation)),$(esc(RList)),$(esc(code)),$r_positive)
    end

    end)
end

# likewise, we define checking intersection points as a macro in order to reduce huge amounts of repeated code
# all are variables.
macro checkIntersectionPoint(point,d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPList,DNList,RPList,RNList,d_ellipse,r_ellipse,d_orientation,r_orientation,code)
    return :(
    if $(esc(code)) == DPRP
        @placeIntersectionPointInLists($(esc(point)), $(esc(d_center)), $(esc(r_center)), $(esc(DVector1)), $(esc(DVector2)), $(esc(RVector1)), $(esc(RVector2)), $(esc(DPList)), $(esc(RPList)), $(esc(d_ellipse)), $(esc(r_ellipse)), $(esc(d_orientation)), $(esc(r_orientation)), $(esc(code)), true, true)
    elseif $(esc(code)) == DPRN
        @placeIntersectionPointInLists($(esc(point)), $(esc(d_center)), $(esc(r_center)), $(esc(DVector1)), $(esc(DVector2)), $(esc(RVector1)), $(esc(RVector2)), $(esc(DPList)), $(esc(RNList)), $(esc(d_ellipse)), $(esc(r_ellipse)), $(esc(d_orientation)), $(esc(r_orientation)), $(esc(code)), true, false)
    elseif $(esc(code)) == DNRP
        @placeIntersectionPointInLists($(esc(point)), $(esc(d_center)), $(esc(r_center)), $(esc(DVector1)), $(esc(DVector2)), $(esc(RVector1)), $(esc(RVector2)), $(esc(DNList)), $(esc(RPList)), $(esc(d_ellipse)), $(esc(r_ellipse)), $(esc(d_orientation)), $(esc(r_orientation)), $(esc(code)), false, true)
    else
        @placeIntersectionPointInLists($(esc(point)), $(esc(d_center)), $(esc(r_center)), $(esc(DVector1)), $(esc(DVector2)), $(esc(RVector1)), $(esc(RVector2)), $(esc(DNList)), $(esc(RNList)), $(esc(d_ellipse)), $(esc(r_ellipse)), $(esc(d_orientation)), $(esc(r_orientation)), $(esc(code)), false, false)
    end
    )
end

macro pushCodeFromSign(PList, NList, point, crossingCode, signCode, positiveTest, negativeTest)
    return :(
        if $(esc(signCode)) == $positiveTest
            push!($(esc(PList)), Intersection($(esc(point))[1], $(esc(point))[2], $crossingCode))
        elseif $(esc(signCode)) == $negativeTest
            push!($(esc(NList)), Intersection($(esc(point))[1], $(esc(point))[2], $crossingCode))
        end
    )
end

macro pushCodeFromSignZero(PList, NList, point, crossingCode, crossingCodeZero, signCode, positiveTest, negativeTest, zeroTest)
    return :(
        if $(esc(signCode)) == $positiveTest
            push!($(esc(PList)), Intersection($(esc(point))[1], $(esc(point))[2], $crossingCode))
        elseif $(esc(signCode)) == $negativeTest
            push!($(esc(NList)), Intersection($(esc(point))[1], $(esc(point))[2], $crossingCode))
        elseif $(esc(signCode)) == $zeroTest
            push!($(esc(PList)), Intersection($(esc(point))[1], $(esc(point))[2], $crossingCodeZero))
            push!($(esc(PList)), Intersection($(esc(point))[1], $(esc(point))[2], $crossingCodeZero))
        end
    )
end

function dot(a::Tuple{Float64,Float64},b::Tuple{Float64,Float64})
    return a[1]*b[1] + a[2]*b[2]
end

# returns a tuple of bools for whether, assuming that the two conic equations cross a boundary defined by vector axis at point, where inside points inside the triangle,
# does the conic equation defined by eq1 cross into the boundary or not
function doesConicEquationCrossDoubleBoundary(eq1::conicEquation, eq2::conicEquation, point::Tuple{Float64,Float64}, axis::Tuple{Float64,Float64}, inside::Tuple{Float64,Float64})
    d1 = tangentDerivative(eq1, point[1], point[2])
    d2 = tangentDerivative(eq2, point[1], point[2])
    grad = gradient(eq2, point[1], point[2])
    if sign(dot(d1,inside))*dot(d1,axis) < sign(dot(d2,inside))*dot(d2,axis)
        return dot(grad,axis) > 0
    else
        return dot(grad,axis) < 0
    end
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

    if ( !eigenvector && abs(d1) > s1 && abs(d2) > s2 && abs(d3) > s3 && ( ( d1 > 0 && d2 > 0 && d3 > 0) || ( d1 < 0 && d2 < 0 && d3 < 0 ) ) ) ||
       ( abs(r1) > s1 && abs(r2) > s2 && abs(r3) > s3 && ( ( r1 > 0 && r2 > 0 && r3 > 0) || ( r1 < 0 && r2 < 0 && r3 < 0 ) ) ) 
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
    d_center = center(DConic)
    r_center = center(RConic)

    d_ellipse = discriminant(DConic) < 0.0
    r_ellipse = discriminant(RConic) < 0.0

    # first line: single point. Second line: single line with no intersection region
    ignore_d = (d_ellipse && evaluate(DConic, d_center[1], d_center[2]) == 0.0) ||
               ( DConicXIntercepts[1] == DConicXIntercepts[2] && DConicXIntercepts[1] != Inf
              && DConicYIntercepts[1] == DConicYIntercepts[2] && DConicYIntercepts[1] != Inf
              && DConicHIntercepts[1] == DConicHIntercepts[2] && DConicHIntercepts[1] != Inf
              && !d_ellipse && !(DConic.A == 0.0 && DConic.B == 0.0 && DConic.C == 0.0)
                )

    ignore_r = (r_ellipse && evaluate(RConic, r_center[1], r_center[2]) == 0.0) ||
                ( RConicXIntercepts[1] == RConicXIntercepts[2] && RConicXIntercepts[1] != Inf
               && RConicYIntercepts[1] == RConicYIntercepts[2] && RConicYIntercepts[1] != Inf
               && RConicHIntercepts[1] == RConicHIntercepts[2] && RConicHIntercepts[1] != Inf
               && !r_ellipse && !(RConic.A == 0.0 && RConic.B == 0.0 && RConic.C == 0.0)
                 )

    # third elt is true for double root, false otherwise
    DPIntercepts = Vector{Intersection}([])
    DNIntercepts = Vector{Intersection}([])
    RPIntercepts = Vector{Intersection}([])
    RNIntercepts = Vector{Intersection}([])

    sizehint!(DPIntercepts, 6)
    sizehint!(RPIntercepts, 6)
    sizehint!(DPIntercepts, 6)
    sizehint!(RNIntercepts, 6)

    # annoying loop unrolling :(
    # we deal with edge-cases where the conics intersect at the boundaries later...
    # note that we do not separate the lists into positive and negative yet!

    if !ignore_d
        if 0 <= DConicXIntercepts[1] <= 1
            class = DCellIntersection(d2, d1, r2, r1, DConicXIntercepts[1])
            if DConicXIntercepts[1] == DConicXIntercepts[2]
                if DConicXIntercepts[1] == 0.0
                    @pushCodeFromSignZero(DPIntercepts, DNIntercepts, (0.0, 0.0), CORNER_13, CORNER_13_Z, class, DP, DN, DZ)
                elseif DConicXIntercepts[1] == 1.0
                    @pushCodeFromSignZero(DPIntercepts, DNIntercepts, (1.0, 0.0), CORNER_12, CORNER_12_Z, class, DP, DN, DZ)
                elseif class == DZ
                    push!(DPIntercepts, Intersection(DConicXIntercepts[1], 0.0, E1Z))
                    push!(DNIntercepts, Intersection(DConicXIntercepts[1], 0.0, E1Z))
                end
            else               
                if DConicXIntercepts[1] == 0.0
                    @pushCodeFromSign(DPIntercepts, DNIntercepts, (0.0, 0.0), CORNER_13, class, DP, DN)
                elseif DConicXIntercepts[1] == 1.0
                    @pushCodeFromSign(DPIntercepts, DNIntercepts, (1.0, 0.0), CORNER_12, class, DP, DN)
                elseif (RConicXIntercepts[1] != DConicXIntercepts[1] && RConicXIntercepts[2] != DConicXIntercepts[1]) || doesConicEquationCrossDoubleBoundary(DConic, RConic, (DConicXIntercepts[1], 0.0), (1.0, 0.0), (0.0, 1.0))
                    @pushCodeFromSign(DPIntercepts, DNIntercepts, (DConicXIntercepts[1], 0.0), E1, class, DP, DN)
                end

                if 0 <= DConicXIntercepts[2] <= 1.0
                    class = DCellIntersection(d2, d1, r2, r1, DConicXIntercepts[2])
                    if DConicXIntercepts[2] == 0.0
                        @pushCodeFromSign(DPIntercepts, DNIntercepts, (DConicXIntercepts[2], 0.0), CORNER_13, class, DP, DN)
                    elseif DConicXIntercepts[2] == 1.0
                        @pushCodeFromSign(DPIntercepts, DNIntercepts, (DConicXIntercepts[2], 0.0), CORNER_12, class, DP, DN)
                    elseif (RConicXIntercepts[1] != DConicXIntercepts[2] && RConicXIntercepts[2] != DConicXIntercepts[1]) || doesConicEquationCrossDoubleBoundary(DConic, RConic, (DConicXIntercepts[2], 0.0), (1.0, 0.0), (0.0, 1.0))
                        @pushCodeFromSign(DPIntercepts, DNIntercepts, (DConicXIntercepts[2], 0.0), E1, class, DP, DN)
                    end
                end
            end
        elseif 0 <= DConicXIntercepts[2] <= 1
            class = DCellIntersection(d2, d1, r2, r1, DConicXIntercepts[2])
            if DConicXIntercepts[2] == 0.0
                @pushCodeFromSign(DPIntercepts, DNIntercepts, (DConicXIntercepts[2], 0.0), CORNER_13, class, DP, DN)
            elseif DConicXIntercepts[2] == 1.0
                @pushCodeFromSign(DPIntercepts, DNIntercepts, (DConicXIntercepts[2], 0.0), CORNER_12, class, DP, DN)
            elseif (RConicXIntercepts[1] != DConicXIntercepts[2] && RConicXIntercepts[2] != DConicXIntercepts[1]) || doesConicEquationCrossDoubleBoundary(DConic, RConic, (DConicXIntercepts[2], 0.0), (1.0, 0.0), (0.0, 1.0))
                @pushCodeFromSign(DPIntercepts, DNIntercepts, (DConicXIntercepts[2], 0.0), E1, class, DP, DN)
            end
        end
    end

    println(DConicXIntercepts)
    println(DPIntercepts)
    println(DNIntercepts)
        exit()

    #     if 0 < DConicYIntercepts[1] <= 1
    #         if 0 < DConicYIntercepts[2] <= 1
    #             if DConicYIntercepts[1] == DConicYIntercepts[2] 
    #                 if DConicYIntercepts[1] == 1.0 || (DConicYIntercepts[1] == d_center[2] && d_center[1] == 0.0)
    #                     push!(DIntercepts, (0.0, 1.0))
    #                 end
    #             else
    #                 push!(DIntercepts, (0.0, DConicYIntercepts[1]))
    #                 push!(DIntercepts, (0.0, DConicYIntercepts[2]))
    #             end
    #         else
    #             push!(DIntercepts, (0.0, DConicYIntercepts[1]))
    #         end
    #     elseif 0 < DConicYIntercepts[2] <= 1
    #         push!(DIntercepts, (0.0, DConicYIntercepts[2]))
    #     end

    #     if 0 < DConicHIntercepts[1] < 1
    #         if 0 < DConicHIntercepts[2] < 1
    #             if DConicHIntercepts[1] == DConicHIntercepts[2]
    #                 if d_center[1] == DConicHIntercepts[1] && d_center[2] == 1.0-DConicHIntercepts[1]
    #                     push!(DIntercepts, (DConicHIntercepts[1], 1.0-DConicHIntercepts[1]))
    #                 end                        
    #             else
    #                 push!(DIntercepts, (DConicHIntercepts[1], 1.0-DConicHIntercepts[1]))
    #                 push!(DIntercepts, (DConicHIntercepts[2], 1.0-DConicHIntercepts[2]))
    #             end
    #         else
    #             push!(DIntercepts, (DConicHIntercepts[1], 1.0-DConicHIntercepts[1]))
    #         end
    #     elseif 0 < DConicYIntercepts[2] < 1
    #         push!(DIntercepts, (DConicHIntercepts[2], 1.0-DConicHIntercepts[2]))
    #     end
    # end

    # if !ignore_r
    #     if 0 <= RConicXIntercepts[1] <= 1
    #         if 0 <= RConicXIntercepts[2] <= 1
    #             if RConicXIntercepts[1] == RConicXIntercepts[2]
    #                 if RConicXIntercepts[1] == 0.0 || RConicXIntercepts[2] == 1.0 || (r_center[1] == RConicXIntercepts[1] && r_center[2] == 0.0)
    #                     push!(RIntercepts, (RConicXIntercepts[1], 0.0))
    #                 end
    #             else
    #                 push!(RIntercepts, (RConicXIntercepts[1], 0.0))
    #                 push!(RIntercepts, (RConicXIntercepts[2], 0.0))
    #             end
    #         else
    #             push!(RIntercepts, (RConicXIntercepts[1], 0.0))
    #         end
    #     elseif 0 <= RConicXIntercepts[2] <= 1
    #         push!(RIntercepts, (RConicXIntercepts[2], 0.0))
    #     end

    #     if 0 < RConicYIntercepts[1] <= 1
    #         if 0 < RConicYIntercepts[2] <= 1
    #             if RConicYIntercepts[1] == RConicYIntercepts[2]
    #                 if RConicYIntercepts[1] == 1.0 || (r_center[2] == RConicYIntercepts[1] && r_center[1] == 0.0)
    #                     push!(RIntercepts, (0.0, 1.0))
    #                 end
    #             else
    #                 push!(RIntercepts, (0.0, RConicYIntercepts[1]))
    #                 push!(RIntercepts, (0.0, RConicYIntercepts[2]))
    #             end
    #         else
    #             push!(RIntercepts, (0.0, RConicYIntercepts[1]))
    #         end
    #     elseif 0 < RConicYIntercepts[2] <= 1
    #         push!(RIntercepts, (0.0, RConicYIntercepts[2]))
    #     end

    #     if 0 < RConicHIntercepts[1] < 1
    #         if 0 < RConicHIntercepts[2] < 1
    #             if RConicHIntercepts[1] == RConicHIntercepts[2]
    #                 if r_center[1] == RConicHIntercepts[1] && r_center[2] == 1.0 - RConicHIntercepts[1]
    #                     push!(RIntercepts, (RConicHIntercepts[1], 1.0-RConicHIntercepts[1]))
    #                 end
    #             else
    #                 push!(RIntercepts, (RConicHIntercepts[1], 1.0-RConicHIntercepts[1]))
    #                 push!(RIntercepts, (RConicHIntercepts[2], 1.0-RConicHIntercepts[2]))
    #             end
    #         else
    #             push!(RIntercepts, (RConicHIntercepts[1], 1.0-RConicHIntercepts[1]))
    #         end
    #     elseif 0 < DConicYIntercepts[2] < 1
    #         push!(RIntercepts, (RConicHIntercepts[2], 1.0-RConicHIntercepts[2]))
    #     end
    # end

    # # Check if either is an internal ellipse
    # # Check that each conic (a) does not not intersect the triangle, (b) is an ellipse, and (c) has a center inside the triangle.
    # # Checking whether or not the conic is an ellipse follows from the sign of the discriminant.
    # d_internal_ellipse = false
    # r_internal_ellipse = false

    # if !ignore_d && length(DIntercepts) == 0 && d_ellipse && (!eigenvector || !(abs(d1) >= s1 && abs(d2) >= s2 && abs(d3) >= s3) )
    #     if is_inside_triangle(d_center[1], d_center[2])
    #         d_internal_ellipse = true
    #     end
    # end

    # if !ignore_r && length(RIntercepts) == 0 && r_ellipse
    #     if is_inside_triangle(r_center[1], r_center[2])
    #         r_internal_ellipse = true
    #     end
    # end

    # # if the two conics do not intersect the triangle, and neither is an internal ellipse,
    # # then the triangle is a standard white triangle.

    # if length(DIntercepts) == 0 && length(RIntercepts) == 0 && !d_internal_ellipse && !r_internal_ellipse
    #     return cellTopologyEigenvalue(vertexTypesEigenvalue, vertexTypesEigenvector, DPArray, DNArray, RPArray, RNArray, RPArrayVec, RNArrayVec)
    # end

    # # If we made it this far, it means that this is an "interesting" case :(

    # # So now we need to intersect the conics with each other. We can find the intersection points
    # # where each individual conic intersects the lines r=d and r=-d.

    # # coefficients for standard form lines of r=0 and d=0 (in the form of ax+by+c=0)
    # a_r = r2-r1
    # b_r = r3-r1
    # c_r = r1
    # a_d = d2-d1
    # b_d = d3-d1
    # c_d = d1

    # # coefficients for r=d in standard form
    # a_rpd = a_r - a_d
    # b_rpd = b_r - b_d
    # c_rpd = c_r - c_d

    # # coefficients for r=-d in standard form
    # a_rnd = a_r + a_d
    # b_rnd = b_r + b_d
    # c_rnd = c_r + c_d

    # # intersect r conic with both of these lines
    # rpd_intersections = intersectWithStandardFormLine(RConic, a_rpd, b_rpd, c_rpd)
    # rnd_intersections = intersectWithStandardFormLine(RConic, a_rnd, b_rnd, c_rnd)

    # rpd_intersection_1 = NULL
    # rpd_intersection_2 = NULL
    # rnd_intersection_1 = NULL
    # rnd_intersection_2 = NULL

    # if is_inside_triangle(rpd_intersections[1][1], rpd_intersections[1][2])
    #     rpd_intersection_1 = DRSignAt(d1, d2, d3, rpd_intersections[1][1], rpd_intersections[1][2], true)
    # end

    # if is_inside_triangle(rpd_intersections[2][1], rpd_intersections[2][2])
    #     rpd_intersection_2 = DRSignAt(d1, d2, d3, rpd_intersections[2][1], rpd_intersections[2][2], true)
    # end

    # if is_inside_triangle(rnd_intersections[1][1], rnd_intersections[1][2])
    #     rnd_intersection_1 = DRSignAt(d1, d2, d3, rnd_intersections[1][1], rnd_intersections[1][2], false)
    # end

    # if is_inside_triangle(rnd_intersections[2][1], rnd_intersections[2][2])
    #     rnd_intersection_2 = DRSignAt(d1, d2, d3, rnd_intersections[2][1], rnd_intersections[2][2], false)
    # end

    # if rpd_intersection_1 == DP
    #     numDP += 1
    #     numRP += 1
    # elseif rpd_intersection_1 == DN
    #     numDN += 1
    #     numRN += 1
    # end

    # if rpd_intersection_2 == DP
    #     numDP += 1
    #     numRP += 1
    # elseif rpd_intersection_2 == DN
    #     numDN += 1
    #     numRN += 1
    # end

    # if rnd_intersection_1 == DP
    #     numDP += 1
    #     numRN += 1
    # elseif rnd_intersection_1 == DN
    #     numDN += 1
    #     numRP += 1
    # end

    # if rnd_intersection_2 == DP
    #     numDP += 1
    #     numRN += 1
    # elseif rnd_intersection_2 == DN
    #     numDN += 1
    #     numRP += 1
    # end

    # # compute the axes that we use for our coordinate transformations
    
    # # first compute the eigenvalues of the hessian, which are useful directions for our purposes
    # eigenRootD = sqrt( 4 * DConic.B^2 + ( 2*DConic.A - 2*DConic.C )^2 )
    # λd2 = (2*DConic.A + 2*DConic.C - eigenRootD) / 2 # we only need the second (negative) eigenvalue.
    # Daxis1x = (λd2 - 2*DConic.C) / DConic.B

    # eigenRootR = sqrt( 4 * RConic.B^2 + ( 2*RConic.A - 2*RConic.C )^2 )
    # λr2 = (2*RConic.A + 2*RConic.C - eigenRootR) / 2
    # Raxis1x = (λr2 - 2*RConic.C) / RConic.B

    # if d_ellipse

    #     # degeneracy warning! (although in that case you would get a parabola)

    #     if Daxis1x > 0
    #         # double check to make sure that this actually always gives the first has positive slope and the second has negative
    #         DVector1 = (Daxis1x, 1.0)
    #         DVector2 = (1.0/Daxis1x, -1.0)
    #     else
    #         DVector1 = (-1.0/Daxis1x, 1.0)
    #         DVector2 = (-Daxis1x, -1.0)
    #     end
    # else
    #     # second eigenvalue not needed here

    #     # axis 1 points left (orient top this way)
    #     # axis 2 points up (in order to orient which curve is on top and thus should be oriented left)

    #     if Daxis1x > 0.0
    #         DVector1 = (-Daxis1x, -1.0)
    #         DVector2 = (-1.0/Daxis1x, 1.0)
    #     else
    #         DVector1 = (Daxis1x, 1.0)
    #         DVector2 = (-1.0/Daxis1x, 1.0)
    #     end

    # end

    # # the same but for r
    # if r_ellipse
    #     if Raxis1x > 0
    #         # double check to make sure that this actually always gives the first has positive slope and the second has negative
    #         RVector1 = (Raxis1x, 1.0)
    #         RVector2 = (1.0/Raxis1x, -1.0)
    #     else
    #         RVector1 = (-1.0/Raxis1x, 1.0)
    #         RVector2 = (-Raxis1x, -1.0)
    #     end
    # else
    #     if Raxis1x > 0.0
    #         RVector1 = (-Raxis1x, -1.0)
    #         RVector2 = (-1.0/Raxis1x, 1.0)
    #     else
    #         RVector1 = (Raxis1x, 1.0)
    #         RVector2 = (-1.0/Raxis1x, 1.0)
    #     end
    # end

    # DPPoints = Vector{Intersection}(undef, 0)
    # DNPoints = Vector{Intersection}(undef, 0)
    # RPPoints = Vector{Intersection}(undef, 0)
    # RNPoints = Vector{Intersection}(undef, 0)

    # sizehint!(DPPoints, numDP)
    # sizehint!(DNPoints, numDN)
    # sizehint!(RPPoints, numRP)
    # sizehint!(RNPoints, numRN)

    # # for hyperbola: 0: not oriented. 1: positive is up. 2: negative is up
    # # not used for ellipse
    # d_orientation = 0
    # r_orientation = 0

    # if d_ellipse
    #     for i in eachindex(DIntercepts)
    #         if DInterceptClasses[i] == DP
    #             push!(DPPoints, ellipse_intersection(DIntercepts[i], d_center, DVector1, DVector2, getCellEdgeFromPoint(DIntercepts[i])))
    #         elseif DInterceptClasses[i] == DN
    #             push!(DNPoints, ellipse_intersection(DIntercepts[i], d_center, DVector1, DVector2, getCellEdgeFromPoint(DIntercepts[i])))
    #         elseif DInterceptClasses[i] == DZ
    #             push!(DPPoints, ellipse_intersection(DIntercepts[i], d_center, DVector1, DVector2, getCellEdgeFromPoint(DIntercepts[i])))
    #             push!(DNPoints, ellipse_intersection(DIntercepts[i], d_center, DVector1, DVector2, getCellEdgeFromPoint(DIntercepts[i])))
    #         end                
    #     end
    # else
    #     # verbose, but I need to reduce the number of checks that happen
    #     for i in eachindex(DIntercepts)
    #         if DInterceptClasses[i] == DP
    #             @orientHyperbolaAndPush(DIntercepts[i],d_center,DVector1,DVector2,d_orientation,DPPoints,getCellEdgeFromPoint(DIntercepts[i]),true)
    #         elseif DInterceptClasses[i] == DN
    #             @orientHyperbolaAndPush(DIntercepts[i],d_center,DVector1,DVector2,d_orientation,DNPoints,getCellEdgeFromPoint(DIntercepts[i]),false)
    #         end
    #     end
    # end

    # if r_ellipse
    #     for i in eachindex(RIntercepts)
    #         if RInterceptClasses[i] == RP
    #             push!(RPPoints, ellipse_intersection(RIntercepts[i], r_center, RVector1, RVector2, getCellEdgeFromPoint(RIntercepts[i])))
    #         elseif RInterceptClasses[i] == RN
    #             push!(RNPoints, ellipse_intersection(RIntercepts[i], r_center, RVector1, RVector2, getCellEdgeFromPoint(RIntercepts[i])))
    #         else
    #             push!(RPPoints, ellipse_intersection(RIntercepts[i], r_center, RVector1, RVector2, getCellEdgeFromPoint(RIntercepts[i])))                
    #             push!(RNPoints, ellipse_intersection(RIntercepts[i], r_center, RVector1, RVector2, getCellEdgeFromPoint(RIntercepts[i])))
    #         end                
    #     end
    # else
    #     # verbose, but I need to reduce the number of checks that happen
    #     for i in eachindex(RIntercepts)
    #         if RInterceptClasses[i] == RP
    #             @orientHyperbolaAndPush(RIntercepts[i],r_center,RVector1,RVector2,r_orientation,RPPoints,getCellEdgeFromPoint(RIntercepts[i]),true)
    #         elseif RInterceptClasses[i] == RN
    #             @orientHyperbolaAndPush(RIntercepts[i],r_center,RVector1,RVector2,r_orientation,RNPoints,getCellEdgeFromPoint(RIntercepts[i]),false)
    #         elseif RInterceptClasses[i] == RZ
    #             @orientHyperbolaAndPush(RIntercepts[i],r_center,RVector1,RVector2,r_orientation,RPPoints,getCellEdgeFromPoint(RIntercepts[i]),true)
    #             @orientHyperbolaAndPush(RIntercepts[i],r_center,RVector1,RVector2,r_orientation,RNPoints,getCellEdgeFromPoint(RIntercepts[i]),false)                
    #         end
    #     end
    # end

    # # then check each of the four intersection points
    # # using macros here actually saved ~200 lines of code (not an exaggeration)
    # if rpd_intersection_1 != NULL
    #     @checkIntersectionPoint(rpd_intersections[1],d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPPoints,DNPoints,RPPoints,RNPoints,d_ellipse,r_ellipse,d_orientation,r_orientation,rpd_intersection_1)
    # end

    # if rpd_intersection_2 != NULL
    #     @checkIntersectionPoint(rpd_intersections[2],d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPPoints,DNPoints,RPPoints,RNPoints,d_ellipse,r_ellipse,d_orientation,r_orientation,rpd_intersection_2)
    # end

    # if rnd_intersection_1 != NULL
    #     @checkIntersectionPoint(rnd_intersections[1],d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPPoints,DNPoints,RPPoints,RNPoints,d_ellipse,r_ellipse,d_orientation,r_orientation,rnd_intersection_1)
    # end

    # if rnd_intersection_2 != NULL
    #     @checkIntersectionPoint(rnd_intersections[2],d_center,r_center,DVector1,DVector2,RVector1,RVector2,DPPoints,DNPoints,RPPoints,RNPoints,d_ellipse,r_ellipse,d_orientation,r_orientation,rnd_intersection_2)
    # end

    # # custom sort function that ensures the clockwise ordering that we were shooting for.
    # sort!(DPPoints)
    # sort!(DNPoints)
    # sort!(RPPoints)
    # sort!(RNPoints)

    # d_tangent_only = true
    # r_tangent_only = true

    # for i in eachindex(DPPoints)
    #     DPArray[i] = DPPoints[i].code
    #     if d_tangent_only && DPArray[i] != E1TANGENT && DPArray[i] != E2TANGENT && DPArray[i] != E3TANGENT
    #         d_tangent_only = false
    #     end
    # end

    # for i in eachindex(DNPoints)
    #     DNArray[i] = DNPoints[i].code
    #     if d_tangent_only && DNArray[i] != E1TANGENT && DNArray[i] != E2TANGENT && DNArray[i] != E3TANGENT
    #         d_tangent_only = false
    #     end        
    # end
    
    # for i in eachindex(RPPoints)
    #     RPArray[i] = RPPoints[i].code
    #     if r_tangent_only && RPArray[i] != E1TANGENT && RPArray[i] != E2TANGENT && RPArray[i] != E3TANGENT
    #         r_tangent_only = false
    #     end        
    # end

    # for i in eachindex(RNPoints)
    #     RNArray[i] = RNPoints[i].code
    #     if r_tangent_only && RNArray[i] != E1TANGENT && RNArray[i] != E2TANGENT && RNArray[i] != E3TANGENT
    #         r_tangent_only = false
    #     end        
    # end

    # if d_ellipse && d_tangent_only
    #     d_center_class = classifyEllipseCenter(d1, d2, d3, r1, r2, r3, d_center[1], d_center[2])
    #     if d_center_class == DP
    #         DPArray[length(DPPoints)+1] = INTERNAL_ELLIPSE
    #     elseif d_center_class == DN
    #         DNArray[length(DNPoints)+1] = INTERNAL_ELLIPSE
    #     end
    # end

    # if r_internal_ellipse
    #     if eigenvector
    #         if (r2-r1)*r_center[1]+(r3-r1)*r_center[2]+r1 >= 0
    #             RPArrayVec[1] = INTERNAL_ELLIPSE
    #         else
    #             RNArrayVec[1] = INTERNAL_ELLIPSE
    #         end
    #     end

    #     if length(RPPoints) == 0 && length(RNPoints) == 0
    #         r_center_class = classifyEllipseCenter(d1,d2,d3,r1,r2,r3,r_center[1],r_center[2])
    #         if r_center_class == RP
    #             RPArray[1] = INTERNAL_ELLIPSE
    #         elseif r_center_class == RN
    #             RNArray[1] = INTERNAL_ELLIPSE
    #         end
    #     end

    # elseif r_ellipse && r_tangent_only
    #     r_center_class = classifyEllipseCenter(d1,d2,d3,r1,r2,r3,r_center[1],r_center[2])
    #     if r_center_class == RP
    #         RPArray[length(RPPoints)+1] = INTERNAL_ELLIPSE
    #     else
    #         RNArray[length(RNPoints)+1] = INTERNAL_ELLIPSE
    #     end
    # end

    # # what a racket
    # return cellTopologyEigenvalue(vertexTypesEigenvalue, vertexTypesEigenvector, DPArray, DNArray, RPArray, RNArray, RPArrayVec, RNArrayVec)
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

    if abs(r1) >= s1 && abs(r2) >= s2 && abs(r3) >= s3 && ( ( r1 >= 0 && r2 >= 0 && r3 >= 0) || ( r1 <= 0 && r2 <= 0 && r3 <= 0 ) )
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

    if 0 <= RConicXIntercepts[1] <= 1
        if RConicXIntercepts[1]*r2 + (1-RConicXIntercepts[1])*r1 > 0
            RPArray[1] += 1
        else
            RNArray[1] += 1
        end
    end

    if 0 <= RConicXIntercepts[2] <= 1
        if RConicXIntercepts[2]*r2 + (1-RConicXIntercepts[2])*r1 > 0
            RPArray[1] += 1
        else
            RNArray[1] += 1
        end
    end

    if 0 < RConicYIntercepts[1] <= 1
        if RConicYIntercepts[1]*r3 + (1-RConicYIntercepts[1])*r1 > 0
            RPArray[3] += 1
        else
            RNArray[3] += 1
        end
    end

    if 0 < RConicYIntercepts[2] <= 1
        if RConicYIntercepts[2]*r3 + (1-RConicYIntercepts[2])*r1 > 0
            RPArray[3] += 1
        else
            RNArray[3] += 1
        end
    end

    if 0 < RConicHIntercepts[1] < 1
        if RConicHIntercepts[1]*r2 + (1-RConicHIntercepts[1])*r3 > 0
            RPArray[2] += 1
        else
            RNArray[2] += 1
        end
    end

    if 0 < RConicHIntercepts[2] < 1
        if RConicHIntercepts[2]*r2 + (1-RConicHIntercepts[2])*r3 > 0
            RPArray[2] += 1
        else
            RNArray[2] += 1
        end
    end

    if RPArray == MArray{Tuple{3}, Int8}(zeros(Int8, 3)) && RNArray == MArray{Tuple{3}, Int8}(zeros(Int8, 3)) && discriminant(RConic) < 0
        if r_center[1] > 0 && r_center[2] > 0 && r_center[2] < 1.0 - r_center[1]
            if (r2-r1)*r_center[1] + (r3-r1)*r_center[2] + r1 > 0
                RPArray[1] = INTERNAL_ELLIPSE
            else
                RNArray[1] = INTERNAL_ELLIPSE
            end
        end
    end

    return cellTopologyEigenvector(vertexTypes, RPArray, RNArray)
end

end