include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")

using ..cellTopology

using StaticArrays

function main()
    d1 = 2.8
    d2 = 3.2
    d3 = 4.2
    w1 = 4.27
    w2 = 0.73
    w3 = 2.9
    r1 = -4.7
    r2 = -1.7
    r3 = 4.5
    θ1 = 4.5
    θ2 = 5.27
    θ3 = 1.5

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    classifyCellEigenvalue(M1,M2,M3)

end

main()