include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")
include("tensorField.jl")

using ..cellTopology
using ..tensorField

using StaticArrays

function main()
    d1 = 7.4
    d2 = -7.6
    d3 = 8.1
    w1 = 3.3
    w2 = 1.53
    w3 = 4.55
    r1 = -3.9
    r2 = -4.3
    r3 = -5.1
    θ1 = 2.2
    θ2 = 3.95
    θ3 = 4.05

    # D = (-11, -2.8, 5.6)
    # W = (4.47, 8.23, 8.3)
    # R = (0, 0, 0)
    # θ = (0, 0, pi)

    # decomp1 = (D[1],R[1],W[1],θ[1])
    # decomp2 = (D[2],R[2],W[2],θ[2])
    # decomp3 = (D[3],R[3],W[3],θ[3])

    # decomp1 = (2.3785250959917903e-7, 0.0, 1.9963058573385585e-7, -1.7316951292458234)
    # decomp2 = (1.2089704071253089e-6, 0.0, 0.0, 0.0)
    # decomp3 = (9.635572130661347e-8, 0.0, 0.0, 0.0)

    # d1, r1, w1, θ1 = decomp1
    # d2, r2, w2, θ2 = decomp2
    # d3, r3, w3, θ3 = decomp3

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    # M1 = SMatrix{2,2,Float64}([-1.5497207641601562e-6 -9.431505532120354e-8; -5.960464477539063e-8 -7.141170499380678e-8])
    # M2 = SMatrix{2,2,Float64}([8.344650268554688e-7 4.697312760981731e-8; 0.0 0.0])
    # M3 = SMatrix{2,2,Float64}([-1.5497207641601562e-6 4.867615643888712e-9; 0.0 -7.141261448850855e-8])

    # top1 = classifyCellEigenvalue(M1,M2,M3,false)
    # println(top1)

    top2 = classifyCellEigenvalue(M1,M2,M3,true)
    println(top2)
    top2 = classifyCellEigenvalue(M1,M2,M3,false)
    println(top2)

end

main()