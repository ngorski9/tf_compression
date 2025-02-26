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

    # D = (5.0, -5.0, 1.0)
    # W = (0.0,0.0,4.6)
    # R = (3.0, -3.0, -2.8)
    # θ = (1.82, 3.48, 4.0)

    # decomp1 = (D[1],R[1],W[1],θ[1])
    # decomp2 = (D[2],R[2],W[2],θ[2])
    # decomp3 = (D[3],R[3],W[3],θ[3])

    decomp1 = (0.0, 0.0, 0.0, 1.82)
    decomp2 = (3.0, 3.6, 2.57, 3.48)
    decomp3 = (3.7, 4.0, 4.6, 4.0)

    d1, r1, w1, θ1 = decomp1
    d2, r2, w2, θ2 = decomp2
    d3, r3, w3, θ3 = decomp3

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    # M1 = SMatrix{2,2,Float64}([5.960464477539063e-8 -1.6746298570069484e-6; 0.0 6.01630745222792e-8])
    # M2 = SMatrix{2,2,Float64}([5.960464477539063e-8 -1.6746453184168786e-6; -5.960464477539063e-8 5.7067154557444155e-8])
    # M3 = SMatrix{2,2,Float64}([5.960464477539063e-8 -1.6777248674770817e-6; 0.0 6.016398401698098e-8])

    top1 = classifyCellEigenvalue(M1,M2,M3,false)
    println(top1)

    top2 = classifyCellEigenvector(M1,M2,M3)
    println(top2)

end

main()