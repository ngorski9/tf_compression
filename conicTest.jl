include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")

using ..cellTopology

using StaticArrays

function main()
    d1 = 1.7
    d2 = 1.6
    d3 = 0.9
    w1 = 9.5
    w2 = 9.5
    w3 = 9.5
    r1 = 0.8
    r2 = 2.2
    r3 = 2.4
    θ1 = 2.75
    θ2 = 1.77
    θ3 = -0.8

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    classifyCellEigenvalue(M1,M2,M3)

end

main()