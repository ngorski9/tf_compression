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

# cell region types
const DP::Int8 = 0
const DN::Int8 = 1
const RP::Int8 = 2
const RN::Int8 = 3
const S::Int8 = 4

# intersection codes
const BLANK::Int8 = 0
const e1::Int8 = 1
const e1P::Int8 = 2
const e1N::Int8 = 3
const e2::Int8 = 4
const e2P::Int8 = 5
const e2N::Int8 = 6
const e3::Int8 = 7
const e3P::Int8 = 8
const e3N::Int8 = 9
const IP::Int8 = 10 # intersect positive
const IN::Int8 = 11 # intersect negative
const INTERIOR_ELLIPSE::Int8 = 12

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

    # store all valid triangle intercepts into fixed size arrays
    numDIntercepts = 0
    numRIntercepts = 0

    DIntercepts = Vector{Tuple{Float64, Float64}}(undef, 6)
    RIntercepts = Vector{Tuple{Float64, Float64}}(undef, 6)

    # annoying loop unrolling :(

    if 0 <= DConicXIntercepts[1] <= 1
        numDIntercepts = 1
        DIntercepts[1] = (DConicXIntercepts[1], 0.0)
    end

    if 0 <= DConicXIntercepts[2] <= 1
        numDIntercepts += 1
        DIntercepts[numDIntercepts] = (DConicXIntercepts[2], 0.0)
    end

    if 0 <= DConicYIntercepts[1] <= 1
        numDIntercepts += 1
        DIntercepts[numDIntercepts] = (0.0,DConicYIntercepts[1])
    end

    if 0 <= DConicYIntercepts[2] <= 1
        numDIntercepts += 1
        DIntercepts[numDIntercepts] = (0.0,DConicYIntercepts[2])
    end

    if 0 <= DConicHIntercepts[1] <= 1
        numDIntercepts += 1
        DIntercepts[numDIntercepts] = (DConicHIntercepts[1], 1-DConicHIntercepts[1])
    end

    if 0 <= DConicHIntercepts[2] <= 1
        numDIntercepts += 1
        DIntercepts[numDIntercepts] = (DConicHIntercepts[2], 1-DConicHIntercepts[2])
    end

    println(numRIntercepts)
    println(RIntercepts)

end

end