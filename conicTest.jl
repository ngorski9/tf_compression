include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")

using ..cellTopology

using StaticArrays

function main()
    d1 = 1.3
    d2 = 0.2
    d3 = 1.0
    w1 = 7.04
    w2 = 6.7
    w3 = 6.72
    r1 = -4.2
    r2 = -6.5
    r3 = 4.9
    θ1 = 4.6
    θ2 = 6.28
    θ3 = -3.0

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    classifyCellEigenvalue(M1,M2,M3)

end

main()