include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")
include("tensorField.jl")

using ..cellTopology
using ..tensorField

using StaticArrays

function main()
    d1 = 1.4
    d2 = 1.7
    d3 = 3.0
    w1 = 1.4
    w2 = 3.63
    w3 = 3.0
    r1 = 0.7
    r2 = 2.2
    r3 = 3.3
    θ1 = 0.0
    θ2 = 2.76
    θ3 = -9.0

    D = (5.0, -5.0, 1.0)
    W = (0.0,0.0,4.6)
    R = (3.0, -3.0, -2.8)
    θ = (1.82, 3.48, 4.0)

    decomp1 = (D[1],R[1],W[1],θ[1])
    decomp2 = (D[2],R[2],W[2],θ[2])
    decomp3 = (D[3],R[3],W[3],θ[3])

    d1, r1, w1, θ1 = decomp1
    d2, r2, w2, θ2 = decomp2
    d3, r3, w3, θ3 = decomp3

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))
    # println(M1)
    # println(M2)
    # println(M3)
    # M1 = SMatrix{2,2,Float64}((-0.02075258642435074, -0.025635547935962677, 0.032959990203380585, 0.031128879636526108))
    # M2 = SMatrix{2,2,Float64}((-0.0158696249127388, -0.010986663401126862, 0.023194067180156708, 0.019531846046447754))
    # M3 = SMatrix{2,2,Float64}((-0.030518509447574615, -0.03662221133708954, 0.03662221133708954, 0.040894802659749985))
    # println(decomposeTensor(M1))
    # println(decomposeTensor(M2))    
    # println(decomposeTensor(M3))
    # println("=============")

    top1 = classifyCellEigenvalue(M1,M2,M3,true)
    println(top1)
    # top2 = classifyCellEigenvector(M1,M2,M3)
    # println(top2)

end

main()