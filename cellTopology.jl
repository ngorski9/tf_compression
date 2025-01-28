module cellTopology

using ..conicUtils

using StaticArrays

export classifyCellEigenvalue

struct cellEigenvalueTopology
    vertexTypes::SArray{Tuple{3}, Int8}
    DPArray::MArray{Tuple{10}, Int8}
    DNArray::MArray{Tuple{10}, Int8}
    RPArray::MArray{Tuple{10}, Int8}
    RNArray::MArray{Tuple{10}, Int8}
end

struct Intersection
    x::Float64
    y::Float64
    code::Int8
end

# intersection codes
const BLANK::Int8 = 0
const e1::Int8 = 1
const e2::Int8 = 2
const e3::Int8 = 3
const DPRP::Int8 = 4 # d positive r positive
const DPRN::Int8 = 5 # d positive r negative
const DNRP::Int8 = 6 # d negative r positive
const DNRN::Int8 = 7 # d negative r negative
const INTERIOR_ELLIPSE::Int8 = 8

# used for specifying that certain intersections are invalid / do not count.
const NULL = -1

# cell region types
const DP::Int8 = 0
const DN::Int8 = 1
const RP::Int8 = 2
const RN::Int8 = 3
const S::Int8 = 4

function classifyTensor(d,r,s)
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

function is_inside_triangle(x::Float64,y::Float64)
    return x >= 0.0 && y >= 0.0 && y <= 1.0-x
end

# returns the signs of d and r when the |d|=s and |r|=s curves intersect.
function DSignAt(d1::Float64, d2::Float64, d3::Float64, x::Float64, y::Float64)
    d = (d2-d1)*x + (d3-d1)*y + d1

    if d > 0
        return DP
    else
        return DN
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
    elseif d > 0
        return DP
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
        return NULL
    elseif r > 0
        return RP
    else
        return RN
    end
end

# While technically we use a barycentric interpolation scheme which is agnostic to the locations of the actual cell vertices,
# for mathematical ease we assume that point 1 is at (0,0), point 2 is at (1,0), and point 3 is at (0,1). Choosing a specific
# embedding will not affect the topology.
function classifyCellEigenvalue(M1::SMatrix{2,2,Float64}, M2::SMatrix{2,2,Float64}, M3::SMatrix{2,2,Float64})
    DPArray = MArray{Tuple{10}, Int8}(zeros(Int8, 10))
    DNArray = MArray{Tuple{10}, Int8}(zeros(Int8, 10))
    RPArray = MArray{Tuple{10}, Int8}(zeros(Int8, 10))
    RNArray = MArray{Tuple{10}, Int8}(zeros(Int8, 10))

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

    vertexTypes = SArray{Tuple{3},Int8}((classifyTensor(d1,r1,s1), classifyTensor(d2,r2,s2), classifyTensor(d3,r3,s3)))

    if ( abs(d1) >= s1 && abs(d2) >= s2 && abs(d3) >= s3 && ( ( d1 >= 0 && d2 >= 0 && d3 >= 0) || ( d1 <= 0 && d2 <= 0 && d3 <= 0 ) ) ) ||
       ( abs(r1) >= s1 && abs(r2) >= s2 && abs(r3) >= s3 && ( ( r1 >= 0 && r2 >= 0 && r3 >= 0) || ( r1 <= 0 && r2 <= 0 && r3 <= 0 ) ) ) 
       # in this case, s is dominated by d or r throughout the entire triangle, so the topology follows from the vertices.
        return cellEigenvalueTopology(vertexTypes, DPArray, DNArray, RPArray, RNArray)
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

    DIntercepts = Vector{Tuple{Float64, Float64}}([])
    RIntercepts = Vector{Tuple{Float64, Float64}}([])

    sizehint!(DIntercepts, 6)
    sizehint!(RIntercepts, 6)

    # annoying loop unrolling :(

    if 0 <= DConicXIntercepts[1] <= 1
        push!(DIntercepts, (DConicXIntercepts[1], 0.0))
    end

    if 0 <= DConicXIntercepts[2] <= 1
        push!(DIntercepts, (DConicXIntercepts[2], 0.0))
    end

    if 0 <= DConicYIntercepts[1] <= 1
        push!(DIntercepts, (0.0, DConicYIntercepts[1]))
    end

    if 0 <= DConicYIntercepts[2] <= 1
        push!(DIntercepts, (0.0, DConicYIntercepts[2]))
    end

    if 0 <= DConicHIntercepts[1] <= 1
        push!(DIntercepts, (DConicHIntercepts[1], 1.0-DConicHIntercepts[1]))
    end

    if 0 <= DConicHIntercepts[2] <= 1
        push!(DIntercepts, (DConicHIntercepts[2], 1.0-DConicHIntercepts[2]))
    end

    if 0 <= RConicXIntercepts[1] <= 1
        push!(RIntercepts, (RConicXIntercepts[1], 0.0))
    end

    if 0 <= RConicXIntercepts[2] <= 1
        push!(RIntercepts, (RConicXIntercepts[2], 0.0))
    end

    if 0 <= RConicYIntercepts[1] <= 1
        push!(RIntercepts, (0.0, RConicYIntercepts[1]))
    end

    if 0 <= RConicYIntercepts[2] <= 1
        push!(RIntercepts, (0.0, RConicYIntercepts[2]))
    end

    if 0 <= RConicHIntercepts[1] <= 1
        push!(RIntercepts, (RConicHIntercepts[1], 1.0-RConicHIntercepts[1]))
    end

    if 0 <= RConicHIntercepts[2] <= 1
        push!(RIntercepts, (RConicHIntercepts[2], 1.0-RConicHIntercepts[2]))
    end

    # Check if either is an internal ellipse
    # Check that each conic (a) does not not intersect the triangle, (b) is an ellipse, and (c) has a center inside the triangle.
    # Checking whether or not the conic is an ellipse follows from the sign of the discriminant.
    d_ellipse = discriminant(DConic) < 0.0
    r_ellipse = discriminant(RConic) < 0.0
    d_internal_ellipse = false
    r_internal_ellipse = false

    if length(DIntercepts) == 0 && d_ellipse
        d_center = center(DConic)
        if is_inside_triangle(d_center[1], d_center[2])
            d_internal_ellipse = true
        end
    end

    if length(RIntercepts) == 0 && r_ellipse
        r_center = center(RConic)
        if is_inside_triangle(r_center[1], r_center[2])
            r_internal_ellipse = true
        end
    end

    # if the two conics do not intersect the triangle, and neither is an internal ellipse,
    # then the triangle is a standard white triangle.

    if length(DIntercepts) == 0 && length(RIntercepts) == 0 && !d_internal_ellipse && !r_internal_ellipse
        return cellEigenvalueTopology(vertexTypes, DPArray, DNArray, RPArray, RNArray)
    end

    # If we made it this far, it means that this is an "interesting" case :(

    # So now we need to intersect the conics with each other. We can find the intersection points
    # where each individual conic intersects the lines r=d and r=-d.

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

    # now we classify and count all intersection types
    DInterceptClasses = Vector{Int8}(undef, length(DIntercepts))
    RInterceptClasses = Vector{Int8}(undef, length(RIntercepts))

    # counts of how many intersections of each type each conic has.
    numDP = 0
    numDN = 0
    numRP = 0
    numRN = 0

    for i in eachindex(DIntercepts)
        intercept_point = DIntercepts[i]
        if intercept_point[2] == 0.0 
            # intersects side 1
            class = DCellIntersection(d2, d1, r2, r1, intercept_point[1])
        elseif intercept_point[1] == 0.0 
            # intersects side 3
            class = DCellIntersection(d3, d1, r3, r1, intercept_point[2])
        else
            # intersects side 2
            class = DCellIntersection(d2, d3, r2, r3, intercept_point[1])
        end

        DInterceptClasses[i] = class

        if class == DP
            numDP += 1
        elseif class == DN
            numDN += 1
        end
    end

    for i in eachindex(RIntercepts)
        intercept_point = RIntercepts[i]
        if intercept_point[2] == 0.0 
            # intersects side 1
            class = RCellIntersection(d2, d1, r2, r1, intercept_point[1])
        elseif intercept_point[1] == 0.0 
            # intersects side 3
            class = RCellIntersection(d3, d1, r3, r1, intercept_point[2])
        else
            # intersects side 2
            class = RCellIntersection(d2, d3, r2, r3, intercept_point[1])
        end

        RInterceptClasses[i] = class

        if class == RP
            numRP += 1
        elseif class == RN
            numRN += 1
        end
    end

    rpd_intersection_1 = NULL
    rpd_intersection_2 = NULL
    rnd_intersection_1 = NULL
    rnd_intersection_2 = NULL

    if is_inside_triangle(rpd_intersections[1][1], rpd_intersections[1][2])
        rpd_intersection_1 = DSignAt(d1, d2, d3, rpd_intersections[1][1], rpd_intersections[1][2])
    end

    if is_inside_triangle(rpd_intersections[2][1], rpd_intersections[2][2])
        rpd_intersection_2 = DSignAt(d1, d2, d3, rpd_intersections[2][1], rpd_intersections[2][2])
    end

    if is_inside_triangle(rnd_intersections[1][1], rnd_intersections[1][2])
        rnd_intersection_1 = DSignAt(d1, d2, d3, rnd_intersections[1][1], rnd_intersections[1][2])
    end

    if is_inside_triangle(rnd_intersections[2][1], rnd_intersections[2][2])
        rnd_intersection_2 = DSignAt(d1, d2, d3, rnd_intersections[2][1], rnd_intersections[2][2])
    end

    if rpd_intersection_1 == DP
        numDP += 1
        numRP += 1
    elseif rpd_intersection_1 == DN
        numDN += 1
        numRN += 1
    end

    if rpd_intersection_2 == DP
        numDP += 1
        numRP += 1
    elseif rpd_intersection_2 == DN
        numDN += 1
        numRN += 1
    end

    if rnd_intersection_1 == DP
        numDP += 1
        numRN += 1
    elseif rnd_intersection_1 == DN
        numDN += 1
        numRP += 1
    end

    if rnd_intersection_2 == DP
        numDP += 1
        numRN += 1
    elseif rnd_intersection_2 == DN
        numDN += 1
        numRP += 1
    end

    # compute the axes that we use for our coordinate transformations
    
    # first compute the eigenvalues of the hessian, which are useful directions for our purposes
    eigenRootD = sqrt( 4 * DConic.B^2 + ( 2*DConic.A - 2*DConic.C )^2 )
    λd2 = (2*DConic.A + 2*DConic.C - eigenRootD) / 2 # we only need the second (negative) eigenvalue.
    Daxis1x = (λd2 - 2*DConic.C) / DConic.B

    eigenRootR = sqrt( 4 * RConic.B^2 + ( 2*RConic.A - 2*RConic.C )^2 )
    λr2 = (2*RConic.A + 2*RConic.C - eigenRootR) / 2
    Raxis1x = (λr2 - 2*RConic.C) / RConic.B

    if d_ellipse

        # degeneracy warning! (although in that case you would get a parabola)

        if Daxis1x > 0
            # double check to make sure that this actually always gives the first has positive slope and the second has negative
            DVector1 = (Daxis1x, 1.0)
            DVector2 = (1.0/Daxis1x, -1.0)
        else
            DVector1 = (-1.0/Daxis1x, 1.0)
            DVector2 = (-Daxis1x, -1.0)
        end
    else
        # second eigenvalue not needed here

        # axis 1 points left (orient top this way)
        # axis 2 points up (in order to orient which curve is on top and thus should be oriented left)

        if Daxis1x > 0.0
            DVector1 = (-Daxis1x, -1.0)
            DVector2 = (-1.0/Daxis1x, 1.0)
        else
            DVector1 = (Daxis1x, 1.0)
            DVector2 = (-1.0/Daxis1x, 1.0)
        end

    end

    # the same but for r
    if r_ellipse
        if Raxis1x > 0
            # double check to make sure that this actually always gives the first has positive slope and the second has negative
            RVector1 = (Raxis1x, 1.0)
            RVector2 = (1.0/Raxis1x, -1.0)
        else
            RVector1 = (-1.0/Raxis1x, 1.0)
            RVector2 = (-Raxis1x, -1.0)
        end
    else
        if Raxis1x > 0.0
            RVector1 = (-Raxis1x, -1.0)
            RVector2 = (-1.0/Raxis1x, 1.0)
        else
            RVector1 = (Raxis1x, 1.0)
            RVector2 = (-1.0/Raxis1x, 1.0)
        end
    end

    DPPoints = Vector{Intersection}(undef, numDP)
    DNPoints = Vector{Intersection}(undef, numDN)
    RPPoints = Vector{Intersection}(undef, numRP)
    RNPoints = Vector{Intersection}(undef, numRN)

    

end

end