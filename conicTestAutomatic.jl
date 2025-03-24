include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")
include("tensorField.jl")

using ..utils
using ..cellTopology
using ..tensorField

using StaticArrays

# decomp1 is the one that matches the expected class.
# the decomps should be tuples of (d,r,s,theta)
function test_corner(decomp1, decomp2, decomp3, expect_val, expect_vec)
    d1, r1, w1, θ1 = decomp1
    d2, r2, w2, θ2 = decomp2
    d3, r3, w3, θ3 = decomp3

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    top1 = classifyCellEigenvector(M1,M2,M3)
    top2 = classifyCellEigenvalue(M1,M2,M3,false)
    top3 = classifyCellEigenvalue(M1,M2,M3,true)

    if top1.vertexTypes[1] != expect_vec
        println(top1.vertexTypes[1])
        println(expect_vec)
        return 1.1
    elseif top2.vertexTypesEigenvalue[1] != expect_val
        println(top2.vertexTypesEigenvalue[1])
        println(expect_val)
        return 1.2
    elseif top3.vertexTypesEigenvalue[1] != expect_val
        return 1.3
    elseif top3.vertexTypesEigenvector[1] != expect_vec
        return 1.4
    end

    top1 = classifyCellEigenvector(M1,M3,M2)
    top2 = classifyCellEigenvalue(M1,M3,M2,false)
    top3 = classifyCellEigenvalue(M1,M3,M2,true)
    if top1.vertexTypes[1] != expect_vec
        return 2.1
    elseif top2.vertexTypesEigenvalue[1] != expect_val
        return 2.2
    elseif top3.vertexTypesEigenvalue[1] != expect_val
        return 2.3
    elseif top3.vertexTypesEigenvector[1] != expect_vec
        return 2.4
    end

    top1 = classifyCellEigenvector(M2,M1,M3)
    top2 = classifyCellEigenvalue(M2,M1,M3,false)
    top3 = classifyCellEigenvalue(M2,M1,M3,true)
    if top1.vertexTypes[2] != expect_vec
        return 3.1
    elseif top2.vertexTypesEigenvalue[2] != expect_val
        return 3.2
    elseif top3.vertexTypesEigenvalue[2] != expect_val
        return 3.3
    elseif top3.vertexTypesEigenvector[2] != expect_vec
        println(top3.vertexTypesEigenvector[2])
        println(expect_vec)
        return 3.4
    end

    top1 = classifyCellEigenvector(M3,M1,M2)
    top2 = classifyCellEigenvalue(M3,M1,M2,false)
    top3 = classifyCellEigenvalue(M3,M1,M2,true)
    if top1.vertexTypes[2] != expect_vec
        return 4.1
    elseif top2.vertexTypesEigenvalue[2] != expect_val
        return 4.2
    elseif top3.vertexTypesEigenvalue[2] != expect_val
        return 4.3
    elseif top3.vertexTypesEigenvector[2] != expect_vec
        return 4.4
    end

    top1 = classifyCellEigenvector(M2,M3,M1)
    top2 = classifyCellEigenvalue(M2,M3,M1,false)
    top3 = classifyCellEigenvalue(M2,M3,M1,true)
    if top1.vertexTypes[3] != expect_vec
        return 5.1
    elseif top2.vertexTypesEigenvalue[3] != expect_val
        return 5.2
    elseif top3.vertexTypesEigenvalue[3] != expect_val
        return 5.3
    elseif top3.vertexTypesEigenvector[3] != expect_vec
        return 5.4
    end

    top1 = classifyCellEigenvector(M3,M2,M1)
    top2 = classifyCellEigenvalue(M3,M2,M1,false)
    top3 = classifyCellEigenvalue(M3,M2,M1,true)
    if top1.vertexTypes[3] != expect_vec
        return 6.1
    elseif top2.vertexTypesEigenvalue[3] != expect_val
        return 6.2
    elseif top3.vertexTypesEigenvalue[3] != expect_val
        return 6.3
    elseif top3.vertexTypesEigenvector[3] != expect_vec
        return 6.4
    end

    return 0.0
end

function test_full_topology(decomp1, decomp2, decomp3, expect::cellTopology.cellTopologyEigenvalue)
    d1, r1, w1, θ1 = decomp1
    d2, r2, w2, θ2 = decomp2
    d3, r3, w3, θ3 = decomp3
    
    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    top1 = classifyCellEigenvector(M1,M2,M3)
    top2 = classifyCellEigenvalue(M1,M2,M3,false)
    top3 = classifyCellEigenvalue(M1,M2,M3,true)

    if top1.vertexTypes != expect.vertexTypesEigenvector
        println(top1.vertexTypes)
        println(expect.vertexTypesEigenvector)
        return 1.1
    end

    if top1.RPArray != expect.RPArrayVec
        println(top1.RPArray)
        println(expect.RPArrayVec)
        return 1.2
    end

    if top1.RNArray != expect.RNArrayVec
        println(top1.RNArray)
        println(expect.RNArrayVec)
        return 1.3
    end

    if top2.vertexTypesEigenvalue != expect.vertexTypesEigenvalue
        println(top2.vertexTypesEigenvalue)
        println(expect.vertexTypesEigenvalue)
        return 2.1
    end
    
    if !cyclicMatch(top2.RPArray,expect.RPArray)
        println(top2.RPArray)
        println(expect.RPArray)
        return 2.2
    end 
    
    if !cyclicMatch(top2.RNArray,expect.RNArray)
        println(top2.RNArray)
        println(expect.RNArray)
        return 2.3
    end

    if !cyclicMatch(top2.DPArray,expect.DPArray)
        println(top2.DPArray)
        println(expect.DPArray)
        return 2.4
    end 
    
    if !cyclicMatch(top2.DNArray,expect.DNArray)
        println(top2.DNArray)
        println(expect.DNArray)
        return 2.5
    end

    if top3.vertexTypesEigenvalue != expect.vertexTypesEigenvalue 
        return 3.1
    end
        
    if top3.vertexTypesEigenvector != expect.vertexTypesEigenvector 
        return 3.2
    end
        
    if !cyclicMatch(top3.RPArray,expect.RPArray)
        return 3.3
    end
        
    if !cyclicMatch(top3.RNArray,expect.RNArray)
        return 3.4
    end
        
    if !cyclicMatch(top3.DPArray,expect.DPArray)
        return 3.5
    end
        
    if !cyclicMatch(top3.DNArray,expect.DNArray)
        return 3.6
    end
        
    if top3.RPArrayVec != expect.RPArrayVec 
        println(top3.RPArrayVec)
        println(expect.RPArrayVec)
        return 3.7
    end
    
    if top3.RNArrayVec != expect.RNArrayVec
        return 3.8
    end

    return 0.0
end

function listsToDecomp(d_list,r_list,s_list,θ_list)
    return [(d_list[1],r_list[1],s_list[1],θ_list[1]),
            (d_list[2],r_list[2],s_list[2],θ_list[2]),
            (d_list[3],r_list[3],s_list[3],θ_list[3])]
end

struct cornerTest
    decomp1::Tuple{Float64,Float64,Float64,Float64}
    decomp2::Tuple{Float64,Float64,Float64,Float64}
    decomp3::Tuple{Float64,Float64,Float64,Float64}
    expect_val::Int8
    expect_vec::Int8
    id::Int64
end

struct fullTest
    decomp1::Tuple{Float64,Float64,Float64,Float64}
    decomp2::Tuple{Float64,Float64,Float64,Float64}
    decomp3::Tuple{Float64,Float64,Float64,Float64}
    expect::cellTopology.cellTopologyEigenvalue
    id::Int64
end

macro add_corner_test(corner_tests, D,R,S,θ, expect_val, expect_vec, id)
    return :(begin
        decomps = listsToDecomp($(esc(D)), $(esc(R)), $(esc(S)), $(esc(θ)))
        push!($(esc(corner_tests)), cornerTest(decomps[1], decomps[2], decomps[3], $(esc(expect_val)), $(esc(expect_vec)), $id))
    end)
end

macro add_full_test(full_tests, D,R,S,θ, expect, id)
    return :(begin
        decomps = listsToDecomp($(esc(D)), $(esc(R)), $(esc(S)), $(esc(θ)))
        push!($(esc(full_tests)), fullTest(decomps[1], decomps[2], decomps[3], $(esc(expect)), $id))
    end)
end

function main()
    corner_tests = []
    full_tests = []

    D = (-1.1,-0.8,0.5)
    W = (1.4,3.63,3.0)
    R = (2.5, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    # sample test; default value
    @add_corner_test(corner_tests, D,R,W,θ,RP,RRP, -1)
    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(RP,S,S),
            SArray{Tuple{3},Int8}(RRP,SRP,SRN),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(3  ,-1 ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(1,0,1),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,0)

    # ------------------------------------------------------------------
    #                      CORNER TYPE TESTS
    # ------------------------------------------------------------------

    D = (-1.1,-0.8,0.5)
    W = (1.4,3.63,3.0)
    R = (-2.5, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,RRN, 1)

    D = (1.6,-0.8,0.5)
    W = (1.4,3.63,3.0)
    R = (1.0, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,SRP, 2)

    D = (-1.6,-0.8,0.5)
    W = (1.4,3.63,3.0)
    R = (1.0, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,SRP, 3)

    D = (1.2,-0.8,0.5)
    W = (1.4,3.63,3.0)
    R = (1.0, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,SRP, 4)

    D = (1.2,-0.8,0.5)
    W = (1.4,3.63,3.0)
    R = (-1.0, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,SRN, 5)

    # ------------------------------------------------------------------
    #                 Single Degenerate Corner Tests
    # ------------------------------------------------------------------

    # DP

    # glances contains
    D = (2.0,-0.5,0.5)
    W = (2.0,2.4,3.0)
    R = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,SRP, 6)

    # glances outside
    D = (2.0,-3.8,-1.7)
    W = (2.0,2.4,3.0)
    R = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,SRP, 7)

    # interior top
    D = (2.0,-3.8,1.8)
    W = (2.0,2.4,3.0)
    R = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,SRP, 8)

    # interior bottom
    D = (2.0,-3.8,1.8)
    W = (2.0,2.4,3.0)
    R = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,SRP, 9)

    # DN

    # glances contains
    D = (-2.0,0.5,-0.5)
    W = (2.0,2.4,3.0)
    R = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,SRP, 10)

    # glances outside
    D = (-2.0,3.8,1.7)
    W = (2.0,2.4,3.0)
    R = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,SRP, 11)

    # interior top
    D = (-2.0,3.8,-1.8)
    W = (2.0,2.4,3.0)
    R = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,SRP, 12)

    # interior bottom
    D = (-2.0,3.8,-1.8)
    W = (2.0,2.4,3.0)
    R = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,SRP, 13)

    # RP

    # glances contains
    R = (2.0,-0.5,0.5)
    W = (2.0,2.4,3.0)
    D = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,DegenRP, 14)

    # glances outside
    R = (2.0,-3.8,-1.7)
    W = (2.0,2.4,3.0)
    D = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,DegenRP, 15)

    # interior top
    R = (2.0,-3.8,1.8)
    W = (2.0,2.4,3.0)
    D = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,DegenRP, 16)

    # interior bottom
    R = (2.0,-3.8,1.8)
    W = (2.0,2.4,3.0)
    D = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,DegenRP, 17)

    # RN

    # glances contains
    R = (-2.0,0.5,-0.5)
    W = (2.0,2.4,3.0)
    D = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,DegenRN, 18)

    # glances outside
    R = (-2.0,3.8,1.7)
    W = (2.0,2.4,3.0)
    D = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,DegenRN, 19)

    # interior top
    R = (-2.0,3.8,-1.8)
    W = (2.0,2.4,3.0)
    D = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,DegenRN, 20)

    # interior bottom
    R = (-2.0,3.8,-1.8)
    W = (2.0,2.4,3.0)
    D = (1.4, 0.4, -0.6)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,DegenRN, 21)

    # ------------------------------------------------------------------
    #                 Double Degenerate Corner Tests
    # ------------------------------------------------------------------

    # --------------------------- DPRP ---------------------------------

    # Both inside (flat edge for D) (very lucky case!)
    D = (2.0,0.8,2.7)
    W = (2.0,2.4,3.0)
    R = (2.0, 0.8, 1.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,DegenRP, 22)

    # Both inside (flat edge for R)
    D = (2.0, 0.8, 1.3)
    W = (2.0,2.4,3.0)
    R = (2.0,0.8,2.7)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,DegenRP, 23)

    # Both inside (unflat edges)
    D = (2.0,0.8,2.7)
    W = (2.0,2.4,3.0)
    R = (2.0, 0.8, 1.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,DegenRP, 24)

    # Both outside
    D = (2.0,-3.5,-3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, -5.7, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,DegenRP, 25)

    # D in R out
    D = (2.0,0.6,-3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, -5.7, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,DegenRP, 26)

    # R in D out
    D = (2.0,-4.8,-3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, 0.0, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,DegenRP, 27)

    # Both in (D trumps R)
    D = (2.0,2.4,-3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, 0.0, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,DegenRP, 28)

    # Both in (same direction but two regions)
    D = (2.0,0.3,-3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, 0.0, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRP,DegenRP, 29)

    # Both in (same direction but no D due to opposite directions)
    D = (2.0,-2.1,0.7)
    W = (2.0,2.4,3.0)
    R = (2.0, 0.0, -2.3)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,DegenRP, 30)

    # both in same direction but overlap due to D opposite direction
    D = (2.0,-2.1,0.7)
    W = (2.0,2.4,3.0)
    R = (2.0, -1.8, -2.3)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRP,DegenRP, 31)

    # opposite directions and dont touch
    D = (2.0,-2.1,0.7)
    W = (2.0,2.4,3.0)
    R = (2.0, -3.6, 4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRP,DegenRP, 32)

    # opposite directions and do touch
    D = (2.0,-2.1,0.7)
    W = (2.0,2.4,3.0)
    R = (2.0, -2.7, 4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRP,DegenRP, 33)

    # D=R and both are greater than S, split
    D = (2.0,-2.1,0.7)
    W = (1.64,2.4,3.0)
    R = (2.0, -2.7, 4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRP,RRP, 34)

    # D=R and both are greater than s, R dominates
    D = (2.0,-2.1,0.7)
    W = (1.64,2.4,3.0)
    R = (2.0, -1.3, 4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,RRP, 35)

    # D=R and both are greater than S, D dominates
    D = (2.0,-2.1,0.7)
    W = (1.64,2.4,3.0)
    R = (2.0, -7.0, -1.9)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,RRP, 36)

    # D=R and both are less than S
    D = (2.0,-2.1,0.7)
    W = (2.64,2.4,3.0)
    R = (2.0, -7.0, -1.9)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,SRP, 37)

    # --------------------------- DPRN ---------------------------------

    # Both inside (flat edge for D) (very lucky case!)
    D = (2.0,0.8,2.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, -0.8, -1.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,DegenRN, 38)

    # Both inside (flat edge for R)
    D = (2.0, 0.8, 1.3)
    W = (2.0,2.4,3.0)
    R = (-2.0,-0.8,-2.7)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,DegenRN, 39)

    # Both inside (unflat edges)
    D = (2.0,0.8,2.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, -0.8, -1.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,DegenRN, 40)

    # Both outside
    D = (2.0,-3.5,-3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 5.7, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,DegenRN, 41)

    # D in R out
    D = (2.0,0.6,-3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 5.7, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,DegenRN, 42)

    # R in D out
    D = (2.0,-4.8,-3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 0.0, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,DegenRN, 43)

    # Both in (D trumps R)
    D = (2.0,2.4,-3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 0.0, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,DegenRN, 44)

    # Both in (same direction but two regions)
    D = (2.0,0.3,-3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 0.0, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRN,DegenRN, 45)

    # Both in (same direction but no D due to opposite directions)
    D = (2.0,-2.1,0.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, 0.0, 2.3)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,DegenRN, 46)

    # both in same direction but overlap due to D opposite direction
    D = (2.0,-2.1,0.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, 1.8, 2.3)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRN,DegenRN, 47)

    # opposite directions and dont touch
    D = (2.0,-2.1,0.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, 3.6, -4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRN,DegenRN, 48)

    # opposite directions and do touch
    D = (2.0,-2.1,0.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, 2.7, -4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRN,DegenRN, 49)

    # D=R and both are greater than S, split
    D = (2.0,-2.1,0.7)
    W = (1.64,2.4,3.0)
    R = (-2.0, 2.7, -4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DPRN,RRN, 50)

    # D=R and both are greater than s, R dominates
    D = (2.0,-2.1,0.7)
    W = (1.64,2.4,3.0)
    R = (-2.0, 1.3, -4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,RRN, 51)

    # D=R and both are greater than S, D dominates
    D = (2.0,-2.1,0.7)
    W = (1.64,2.4,3.0)
    R = (-2.0, 7.0, 1.9)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DP,RRN, 52)

    # D=R and both are less than S
    D = (2.0,-2.1,0.7)
    W = (2.64,2.4,3.0)
    R = (-2.0, 7.0, 1.9)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,SRN, 53)

        # --------------------------- DNRN ---------------------------------

    # Both inside (flat edge for D) (very lucky case!)
    D = (-2.0,-0.8,-2.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, -0.8, -1.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,DegenRN, 54)

    # Both inside (flat edge for R)
    D = (-2.0, -0.8, -1.3)
    W = (2.0,2.4,3.0)
    R = (-2.0,-0.8,-2.7)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,DegenRN, 55)

    # Both inside (unflat edges)
    D = (-2.0,-0.8,-2.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, -0.8, -1.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,DegenRN, 56)

    # Both outside
    D = (-2.0,3.5,3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 5.7, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,DegenRN, 57)

    # D in R out
    D = (-2.0,-0.6,3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 5.7, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,DegenRN, 58)

    # R in D out
    D = (-2.0,4.8,3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 0.0, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,DegenRN, 59)

    # Both in (D trumps R)
    D = (-2.0,-2.4,3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 0.0, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,DegenRN, 60)

    # Both in (same direction but two regions)
    D = (-2.0,-0.3,3.6)
    W = (2.0,3.0,2.0)
    R = (-2.0, 0.0, 2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRN,DegenRN, 61)

    # Both in (same direction but no D due to opposite directions)
    D = (-2.0,2.1,-0.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, 0.0, 2.3)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,DegenRN, 62)

    # both in same direction but overlap due to D opposite direction
    D = (-2.0,2.1,-0.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, 1.8, 2.3)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRN,DegenRN, 63)

    # opposite directions and dont touch
    D = (-2.0,2.1,-0.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, 3.6, -4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRN,DegenRN, 64)

    # opposite directions and do touch
    D = (-2.0,2.1,-0.7)
    W = (2.0,2.4,3.0)
    R = (-2.0, 2.7, -4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRN,DegenRN, 65)

    # D=R and both are greater than S, split
    D = (-2.0,2.1,-0.7)
    W = (1.64,2.4,3.0)
    R = (-2.0, 2.7, -4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRN,RRN, 66)

    # D=R and both are greater than s, R dominates
    D = (-2.0,2.1,-0.7)
    W = (1.64,2.4,3.0)
    R = (-2.0, 1.3, -4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RN,RRN, 67)

    # D=R and both are greater than S, D dominates
    D = (-2.0,2.1,-0.7)
    W = (1.64,2.4,3.0)
    R = (-2.0, 7.0, 1.9)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,RRN, 68)

    # D=R and both are less than S
    D = (-2.0,2.1,-0.7)
    W = (2.64,2.4,3.0)
    R = (-2.0, 7.0, 1.9)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,SRN, 69)

        # --------------------------- DNRP ---------------------------------

    # Both inside (flat edge for D) (very lucky case!)
    D = (-2.0,-0.8,-2.7)
    W = (2.0,2.4,3.0)
    R = (2.0, 0.8, 1.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,DegenRP, 70)

    # Both inside (flat edge for R)
    D = (-2.0, -0.8, -1.3)
    W = (2.0,2.4,3.0)
    R = (2.0,0.8,2.7)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,DegenRP, 71)

    # Both inside (unflat edges)
    D = (-2.0,-0.8,-2.7)
    W = (2.0,2.4,3.0)
    R = (2.0, 0.8, 1.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,DegenRP, 72)

    # Both outside
    D = (-2.0,3.5,3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, -5.7, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,DegenRP, 73)

    # D in R out
    D = (-2.0,-0.6,3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, -5.7, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,DegenRP, 74)

    # R in D out
    D = (-2.0,4.8,3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, 0.0, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,DegenRP, 75)

    # Both in (D trumps R)
    D = (-2.0,-2.4,3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, 0.0, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,DegenRP, 76)

    # Both in (same direction but two regions)
    D = (-2.0,-0.3,3.6)
    W = (2.0,3.0,2.0)
    R = (2.0, 0.0, -2.3)
    θ = (3.72, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRP,DegenRP, 77)

    # Both in (same direction but no D due to opposite directions)
    D = (-2.0,2.1,-0.7)
    W = (2.0,2.4,3.0)
    R = (2.0, 0.0, -2.3)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,DegenRP, 78)

    # both in same direction but overlap due to D opposite direction
    D = (-2.0,2.1,-0.7)
    W = (2.0,2.4,3.0)
    R = (2.0, -1.8, -2.3)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRP,DegenRP, 79)

    # opposite directions and dont touch
    D = (-2.0,2.1,-0.7)
    W = (2.0,2.4,3.0)
    R = (2.0, -3.6, 4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRP,DegenRP, 80)

    # opposite directions and do touch
    D = (-2.0,2.1,-0.7)
    W = (2.0,2.4,3.0)
    R = (2.0, -2.7, 4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRP,DegenRP, 81)

    # D=R and both are greater than S, split
    D = (-2.0,2.1,-0.7)
    W = (1.64,2.4,3.0)
    R = (2.0, -2.7, 4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DNRP,RRP, 82)

    # D=R and both are greater than s, R dominates
    D = (-2.0,2.1,-0.7)
    W = (1.64,2.4,3.0)
    R = (2.0, -1.3, 4.8)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,RP,RRP, 83)

    # D=R and both are greater than S, D dominates
    D = (-2.0,2.1,-0.7)
    W = (1.64,2.4,3.0)
    R = (2.0, -7.0, -1.9)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,DN,RRP, 84)

    # D=R and both are less than S
    D = (-2.0,2.1,-0.7)
    W = (2.64,2.4,3.0)
    R = (2.0, -7.0, -1.9)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,S,SRP, 85)

    # ---------------------- zero corner --------------------------

    D = (0.0,2.1,-0.7)
    W = (0.0,2.4,3.0)
    R = (0.0, -7.0, -1.9)
    θ = (3.08, 5.9, 2.0)

    @add_corner_test(corner_tests, D,R,W,θ,Z,Z, 86)

    # ------------------------------------------------------------------
    #                 some normal cases (sanity check)
    # ------------------------------------------------------------------

    # D dominates everything
    D = (2.0,2.9,3.2)
    W = (1.64,2.4,3.0)
    R = (0.5, 0.0, -2.1)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,DP,DP),
            SArray{Tuple{3},Int8}(SRP,SYM,SRN),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,2,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,87)

    # R dominates everything
    D = (0.5, 0.0, -2.1)
    W = (1.64,2.4,3.0)
    R = (2.0,2.9,3.2)    
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(RP,RP,RP),
            SArray{Tuple{3},Int8}(RRP,RRP,RRP),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,88)

    # D dominates everything but intersects a corner
    D = (1.64,2.9,3.2)
    W = (1.64,2.4,3.0)
    R = (0.5, 0.0, -2.1)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,DP,DP),
            SArray{Tuple{3},Int8}(SRP,SYM,SRN),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,2,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,89)

    # A normal test - two hyperbolas
    D = (2.1,-3.1,3.2)
    W = (1.64,2.4,3.0)
    R = (0.5, 0.0, -2.1)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,DN,DP),
            SArray{Tuple{3},Int8}(SRP,SYM,SRN),
            MArray{Tuple{10},Int8}(E2ClosestLow,DPRN,-DPRN,-1  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(1  ,-E2ClosestHigh ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,DPRN,-DPRN,-2  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,2,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,90)

    # A normal test - ellipse hyperbola
    D = (2.1,-1.2,3.2)
    W = (1.64,2.4,3.0)
    R = (0.5, 0.0, -2.1)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,S,DP),
            SArray{Tuple{3},Int8}(SRP,SYM,SRN),
            MArray{Tuple{10},Int8}(-DPRN, -1,0 ,0 ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,DPRN,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,2,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,91)

    # A normal test - ellipse hyperbola
    D = (2.1,-1.2,3.2)
    W = (1.64,2.4,3.0)
    R = (0.5, 2.1, -2.1)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,S,DP),
            SArray{Tuple{3},Int8}(SRP,SRP,SRN),
            MArray{Tuple{10},Int8}(2, DPRP ,0 ,0 ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-DPRP  ,-1  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(2,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,92)

    # A normal test - hyperbola ellipse

    # A normal test - ellipse hyperbola
    D = (2.1,-1.2,3.2)
    W = (1.64,2.4,3.0)
    R = (0.3, 2.6, -6.0)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,RP,RN),
            SArray{Tuple{3},Int8}(SRP,RRP,RRN),
            MArray{Tuple{10},Int8}(-DPRN, DPRP ,0 ,0 ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-DPRP  ,-2  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,DPRN,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(1,1,0),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,93)

    # interior ellipse intersects interior hyperbola
    D = (0.7,-0.7,2.2)
    W = (1.64,2.4,3.0)
    R = (0.3, 2.6, -6.0)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,RP,RN),
            SArray{Tuple{3},Int8}(SRP,RRP,RRN),
            MArray{Tuple{10},Int8}(DPRP ,-DPRP ,DPRN , -DPRN, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(1  ,DPRP  ,-DPRP  ,-2  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,DPRN,-DPRN,-3  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(1,1,0),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,94)

    # Two overlapping interior ellipses
    D = (0.7,-0.7,2.2)
    W = (1.64,2.4,3.0)
    R = (0.2, 0.3, 0.6)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,S,S),
            SArray{Tuple{3},Int8}(SRP,SRP,SRP),
            MArray{Tuple{10},Int8}(DPRP ,-DPRP ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-DPRP  ,DPRP  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(INTERNAL_ELLIPSE,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,95)

    # R internal ellipse is eclipsed by D internal ellipse
    D = (0.7,-0.4,2.2)
    W = (1.64,2.4,3.0)
    R = (0.2, 0.3, 0.6)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,S,S),
            SArray{Tuple{3},Int8}(SRP,SRP,SRP),
            MArray{Tuple{10},Int8}(INTERNAL_ELLIPSE ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(INTERNAL_ELLIPSE,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,96)

    # ------------------------------------------------------------------
    #                 degenerate full cases
    # ------------------------------------------------------------------

    # D dominates R internal ellipse. Glances corner but doesn't enter there.
    D = (0.7,2.4,2.2)
    W = (1.64,2.4,3.0)
    R = (0.2, 0.3, 0.6)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,DP,S),
            SArray{Tuple{3},Int8}(SRP,SRP,SRP),
            MArray{Tuple{10},Int8}(1 ,-2 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(INTERNAL_ELLIPSE,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,97)

    # DP glances corner 13 and does not enter there.
    # DN enters from corner 23
    D = (-2.5,2.4,-3.0)
    W = (1.64,2.4,3.0)
    R = (0.2, 0.3, 0.6)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DN,S,DN),
            SArray{Tuple{3},Int8}(SRP,SRP,SRP),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(CORNER_23  ,DNRP  ,-DNRP  ,-1  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-DNRP  ,DNRP  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(INTERNAL_ELLIPSE,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,98)

    # DP crosses corners 1 and 2. DN glances but doesn't cross corner 3
    D = (1.64,2.4,-3.0)
    W = (1.64,2.4,3.0)
    R = (0.2, 0.3, 0.6)
    θ = (3.08, 5.9, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,DP,S),
            SArray{Tuple{3},Int8}(SRP,SRP,SRP),
            MArray{Tuple{10},Int8}(CORNER_13 ,-CORNER_12 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(INTERNAL_ELLIPSE,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,99)

    # DP parallels edge 1. DN glances corner 3. R intersects nothing.
    D = (1.64,2.4,-3.0)
    W = (1.64,2.4,3.0)
    R = (0.2, -2.0, 0.6)
    θ = (0.0, 0.0, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,S,S),
            SArray{Tuple{3},Int8}(SRP,SRN,SRP),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,100)

    # DP parallels edge 1. DN glances corner 3. R intersects twice
    D = (1.64,2.4,-3.0)
    W = (1.64,2.4,3.0)
    R = (2.1, -3.3, 0.6)
    θ = (0.0, 0.0, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(RP,RN,S),
            SArray{Tuple{3},Int8}(RRP,RRN,SRP),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(3  ,-1  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(1  ,-2,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(1,0,1),
            MArray{Tuple{3},Int8}(1,1,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,101)

    # RP parallels edge 1. RN glances corner 3. D doesn't intersect
    D = (1.2,1.1,-2.7)
    W = (1.64,2.4,3.0)
    R = (1.64, 2.4, -3.0)
    θ = (0.0, 0.0, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,S,S),
            SArray{Tuple{3},Int8}(DegenRP,DegenRP,DegenRN),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(STRAIGHT_ANGLES,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,102)

    # RP parallels edge 1. RN glances corner 3. D intersects twice
    D = (1.2,3.0,-3.6)
    W = (1.64,2.4,3.0)
    R = (1.64, 2.4, -3.0)
    θ = (0.0, 0.0, 2.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,DP,DN),
            SArray{Tuple{3},Int8}(DegenRP,DegenRP,DegenRN),
            MArray{Tuple{10},Int8}(1 ,-2 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,-3  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(STRAIGHT_ANGLES,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,103)

    # RP parallels edge 1. DP parallels edge 2.
    D = (1.2,2.4,3.0)
    W = (1.64,2.4,3.0)
    R = (1.64, 2.4, -3.0)
    θ = (0.0, 0.0, 0.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,S,S),
            SArray{Tuple{3},Int8}(DegenRP,DegenRP,DegenRN),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(STRAIGHT_ANGLES,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,104)

    # RP and DP both parallel edge 1 but aren't identical
    D = (1.64,2.4,0.9)
    W = (1.64,2.4,3.0)
    R = (1.64, 2.4, -3.0)
    θ = (0.0, 0.0, 0.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,S,S),
            SArray{Tuple{3},Int8}(DegenRP,DegenRP,DegenRN),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(STRAIGHT_ANGLES,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,-104)

    # RP and DP both parallel edge 1 but now D dominates the entire triangle
    D = (1.64,2.4,4.5)
    W = (1.64,2.4,3.0)
    R = (1.64, 2.4, -3.0)
    θ = (0.0, 0.0, 0.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,DP,DP),
            SArray{Tuple{3},Int8}(DegenRP,DegenRP,DegenRN),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(STRAIGHT_ANGLES,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,-105)

    # R and D are exactly equal.
    D = (2.7, 3.8, -3.4)
    W = (1.64,2.4,3.0)
    R = (2.7, 3.8, -3.4)
    θ = (1.8, 2.48, 1.4)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(RP,RP,RN),
            SArray{Tuple{3},Int8}(RRP,RRP,RRN),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(3  ,-2  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,-3,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,105)

    # R and D are exactly equal and pass through corners
    D = (1.64, 2.4, -3.0)
    W = (1.64,2.4,3.0)
    R = (1.64, 2.4, -3.0)
    θ = (1.8, 2.48, 1.4)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(RP,RP,S),
            SArray{Tuple{3},Int8}(DegenRP,DegenRP,DegenRN),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(CORNER_13  , -CORNER_12  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,106)

    # R and D intersect at bottom edge. D enters but not R
    D = (-3.0, -0.1, -5.3)
    W = (1.64,2.4,3.0)
    R = (3.0, 0.1, -5.8)
    θ = (0.0, 0.0, 5.1)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DN,S,RN),
            SArray{Tuple{3},Int8}(RRP,SRP,RRN),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-DNRN  ,-1  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,DNRN,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(1,0,1),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,107)

    # R and D intersect at bottom edge. R enters but not D
    D = (-3.0, -0.1, 8.1)
    W = (1.64,2.4,3.0)
    R = (3.0, 0.1, -5.8)
    θ = (0.0, 0.0, 5.1)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(RP,S,DP),
            SArray{Tuple{3},Int8}(RRP,SRP,RRN),
            MArray{Tuple{10},Int8}(2 ,-3 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(3  , -1  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(1,0,1),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,108)

    # R and D intersect at bottom edge. Neither enters.
    # (this one is REALLY weird)
    D = (5.0, -9.0, -5.4)
    W = (2.0,2.0,4.6)
    R = (2.5, 1.5, 4.9)
    θ = (0.0, 0.0, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,DN,DN),
            SArray{Tuple{3},Int8}(RRP,SRP,RRP),
            MArray{Tuple{10},Int8}(0 ,0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(INTERNAL_ELLIPSE  , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(1,1,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,109)

    # R and D intersect at bottom edge. Both enter.
    D = (5.0, -9.0, 6.8)
    W = (2.0,2.0,4.6)
    R = (2.5, 1.5, -4.8)
    θ = (0.0, 0.0, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,DN,DP),
            SArray{Tuple{3},Int8}(RRP,SRP,RRN),
            MArray{Tuple{10},Int8}(-DPRN ,DPRP ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(1  ,-2  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-DPRP  , -1  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,DPRN,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(1,0,1),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,110)

    # R and D intersect at a corner. D enters but not R
    D = (2.0, 7.6, -5.5)
    W = (2.0,2.0,4.6)
    R = (2.0, 1.5, -4.8)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,DP,DN),
            SArray{Tuple{3},Int8}(DegenRP,SRP,RRN),
            MArray{Tuple{10},Int8}(CORNER_13, -2 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,-3  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(1,0,0),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,111)

    # R and D intersect at a corner. R enters but not D
    D = (2.0, 7.6, -5.5)
    W = (2.0,2.0,4.6)
    R = (2.0, 8.3, -4.8)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(RP,RP,DN),
            SArray{Tuple{3},Int8}(DegenRP,RRP,RRN),
            MArray{Tuple{10},Int8}(0, 0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,-3  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(CORNER_13  , -2  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,1,0),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,112)

    # R and D intersect at a corner. Neither enters
    D = (2.0, 7.6, -5.5)
    W = (2.0,2.0,4.6)
    R = (2.0, -3.3, 4.9)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DPRP,DP,DN),
            SArray{Tuple{3},Int8}(DegenRP,RRN,RRP),
            MArray{Tuple{10},Int8}(-DPRP, -2 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  ,DNRP  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-DNRP  , DPRP  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  ,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,1,0),
            MArray{Tuple{3},Int8}(1,1,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,113)
    
    # R and D intersect at a corner. Both enter
    D = (2.0, 3.0, -5.5)
    W = (2.0,2.0,4.6)
    R = (2.0, -3.3, -1.1)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DPRP,RN,DN),
            SArray{Tuple{3},Int8}(DegenRP,RRN,SRN),
            MArray{Tuple{10},Int8}(CORNER_13, DPRN ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(2  , -3  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(3  , -CORNER_13  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-DPRN  , -2,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,0,1),
            MArray{Tuple{3},Int8}(1,1,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,114)

    # One corner is zero
    D = (0.0, 3.0, 3.7)
    W = (0.0,2.57,4.6)
    R = (0.0, 3.6, 4.0)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(Z,RP,S),
            SArray{Tuple{3},Int8}(Z,RRP,SRP),
            MArray{Tuple{10},Int8}(CORNER_13_Z, 0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(CORNER_13_Z  , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(CORNER_13_Z  , -2  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(CORNER_13_Z  , 0 ,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,1,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,115)

    # Two corners are zero
    D = (0.0, 0.0, 3.7)
    W = (0.0,0.0,4.6)
    R = (0.0, 0.0, 5.4)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(Z,Z,RP),
            SArray{Tuple{3},Int8}(Z,Z,RRP),
            MArray{Tuple{10},Int8}(0, 0,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0 , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0 ,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,116)

    # Two corners are zero (again)
    D = (0.0, 0.0, 10.0)
    W = (0.0,0.0,4.6)
    R = (0.0, 0.0, 5.4)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(Z,Z,DP),
            SArray{Tuple{3},Int8}(Z,Z,RRP),
            MArray{Tuple{10},Int8}(0, 0,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0 , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0 ,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,117)

    # Two corners are zero (again again)
    D = (0.0, 0.0, 10.0)
    W = (0.0,0.0,4.6)
    R = (0.0, 0.0, 1.2)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(Z,Z,DP),
            SArray{Tuple{3},Int8}(Z,Z,SRP),
            MArray{Tuple{10},Int8}(0, 0,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0 , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0 ,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,118)

    # all corners are zero
    D = (0.0, 0.0, 0.0)
    W = (0.0,0.0,0.0)
    R = (0.0, 0.0, 0.0)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(Z,Z,Z),
            SArray{Tuple{3},Int8}(Z,Z,Z),
            MArray{Tuple{10},Int8}(0, 0,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0 , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0  , 0 ,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,119)

    # zero along an edge
    D = (5.0, -5.0, 1.0)
    W = (0.0,0.0,4.6)
    R = (3.0, -3.0, -2.8)
    θ = (1.82, 3.48, 4.0)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DP,DN,S),
            SArray{Tuple{3},Int8}(RRP,RRN,SRN),
            MArray{Tuple{10},Int8}(3, E1Z ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(E1Z , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(E1Z  , 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(E1Z  , -2 ,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{3},Int8}(1,0,1),
            MArray{Tuple{3},Int8}(1,1,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    ,120)

    # ambiguous two ellipse case.
    D = (-3.0, -9.0, -4.0)
    W = (10.0, 10.0, 4.55)
    R = (-9.0, -3.0, -1.0)
    θ = (2.2, 3.95, 4.05)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,S,S),
            SArray{Tuple{3},Int8}(SRN,SRN,SRN),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-E1ClosestHigh, 1 ,-3 , E3ClosestHigh, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-1, E1ClosestLow ,-E3ClosestLow, 3  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(2,0,2),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 121)

    # everything is white (idk why I didnt test this yet)
    D = (-6.5, -2.7, -2.1)
    W = (10, 10, 9.85)
    R = (-5.4, -3.0, -1.0)
    θ = (2.2, 3.95, 4.05)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(S,S,S),
            SArray{Tuple{3},Int8}(SRN,SRN,SRN),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 122)

    # corners are R, but the region in the middle is D-
    D = (-3.7, -2.8, -3.6)
    W = (4.47, 3.5, 4.8)
    R = (-6.2, -5.3, 6.6)
    θ = (2.2, 2.1, 4.05)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(RN,RN,RP),
            SArray{Tuple{3},Int8}(RRN,RRN,RRP),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(INTERNAL_ELLIPSE, 0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 123)

    # corners are R, but the region in the middle is D+
    D = (3.7, 2.8, 3.6)
    W = (4.47, 3.5, 4.8)
    R = (-6.2, -5.3, 6.6)
    θ = (2.2, 2.1, 4.05)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(RN,RN,RP),
            SArray{Tuple{3},Int8}(RRN,RRN,RRP),
            MArray{Tuple{10},Int8}(INTERNAL_ELLIPSE, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Int8}(0,1,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 124)

    # corners are D but the region in the middle is R-
    D = (-6.2, -5.3, 6.6)
    W = (4.47, 3.5, 4.8)
    R = (-3.7, -2.8, -3.6)
    θ = (2.2, 2.1, 4.05)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DN,DN,DP),
            SArray{Tuple{3},Int8}(SRN,SRN,SRN),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(INTERNAL_ELLIPSE, 0 ,0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Int8}(0,2,2),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 125)

    # corners are D but the region in the middle is R+
    D = (-6.2, -5.3, 6.6)
    W = (4.47, 3.5, 4.8)
    R = (3.7, 2.8, 3.6)
    θ = (2.2, 2.1, 4.05)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DN,DN,DP),
            SArray{Tuple{3},Int8}(SRP,SRP,SRP),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(INTERNAL_ELLIPSE, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,2,2),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 126)

    # similar to before, but the values for R aren't all positive, so the type cant be inferred from the vertices.
    D = (-6.2, -5.3, 6.6)
    W = (4.47, 3.5, 4.8)
    R = (0.0, -0.3, 4.0)
    θ = (0.37, 1.17, 3.84)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DN,DN,DP),
            SArray{Tuple{3},Int8}(SYM,SRN,SRP),
            MArray{Tuple{10},Int8}(0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(INTERNAL_ELLIPSE, 0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0, 0 ,0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,2,2),
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 127)

    # zero in the middle
    D = (-10, -1.8, 6.6)
    W = (4.47, 8.23, 8.3)
    R = (-7.1945144, 7.1, -1.3)
    θ = (0, 0, pi)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DN,S,S),
            SArray{Tuple{3},Int8}(RRN,SRP,SRN),
            MArray{Tuple{10},Int8}(Z, -E2  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(Z, -E1 ,0 , 0, 0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(E2, Z  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(E3, Z ,0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,2,0), # we'll need to change this later!
            MArray{Tuple{3},Int8}(1,0,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 128)

    # two double lines that don't overlap.
    D = (-11, -2.8, 5.6)
    W = (4.47, 8.23, 8.3)
    R = (-7.1945144, 7.1, -1.3)
    θ = (0, 0, pi)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DN,S,S),
            SArray{Tuple{3},Int8}(RRN,SRP,SRN),
            MArray{Tuple{10},Int8}(-DPRP, -E2  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(-DNRN, DNRP, -DNRP, -E1 ,0 , 0, 0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(E2, DNRP  ,-DNRP  ,DPRP  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(E3, DNRN ,0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,2,0), # we'll need to change this later!
            MArray{Tuple{3},Int8}(1,0,1),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 129)

    # two double lines. Red is zero so it doesn't interact with D at all.
    D = (-11, -2.8, 5.6)
    W = (4.47, 8.23, 8.3)
    R = (0.0, 0.0, 0.0)
    θ = (0, 0, pi)

    @add_full_test(full_tests, D,R,W,θ,
        cellTopology.cellTopologyEigenvalue(
            MArray{Tuple{3},Int8}(DN,S,S),
            SArray{Tuple{3},Int8}(SYM,SYM,SYM),
            MArray{Tuple{10},Int8}(E2, Z, -E2  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(E3, Z, -E1 , 0 ,0 , 0, 0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0,0,0,0  ,0  ,0  ,0  ,0  ,0  ,0  ),
            MArray{Tuple{10},Int8}(0,0 ,0, 0  ,0  ,0  ,0  ,0  ,0  ,0  ),            
            MArray{Tuple{3},Int8}(0,0,0), # we'll need to change this later!
            MArray{Tuple{3},Int8}(0,0,0),
            MArray{Tuple{3},Bool}(false,false,false)
        )
    , 130)

    # ------------------------------------------------------------------
    #                 end of automated tests
    # ------------------------------------------------------------------

    for c in corner_tests
        test_output = test_corner(c.decomp1, c.decomp2, c.decomp3, c.expect_val, c.expect_vec)
        if test_output != 0.0
            println("corner test $(c.id) returned $test_output")
            println(c.decomp1)
            println(c.decomp2)
            println(c.decomp3)
            println(c.expect_val)
            println(c.expect_vec)
            exit()
        end
    end

    for f in full_tests
        test_output = test_full_topology(f.decomp1, f.decomp2, f.decomp3, f.expect)
        println(f.id)
        if test_output != 0.0
            println("full test $(f.id) returned $test_output")
            println(f.decomp1)
            println(f.decomp2)
            println(f.decomp3)
            println(f.expect)
            exit()
        end
    end

    println("all tests passed")
end

main()