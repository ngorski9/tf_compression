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

    D = (-10, -1.8, 6.6)
    W = (4.47, 8.23, 8.3)
    R = (-7.1945144, 7.1, -1.3)
    θ = (0, 0, pi)

    decomp1 = (D[1],R[1],W[1],θ[1])
    decomp2 = (D[2],R[2],W[2],θ[2])
    decomp3 = (D[3],R[3],W[3],θ[3])

    # decomp1 = (-3.0, -9.0, 10.0, 2.2)
    # decomp2 = (-9.0, -3.0, 10.0, 3.95)
    # decomp3 = (-4.0, -1.0, 4.55, 4.05)

    d1, r1, w1, θ1 = decomp1
    d2, r2, w2, θ2 = decomp2
    d3, r3, w3, θ3 = decomp3

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

end

main()