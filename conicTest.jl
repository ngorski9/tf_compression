include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")

using ..cellTopology

using StaticArrays

function main()
    d1 = 2.3
    d2 = -0.6
    d3 = 1.3
    w1 = 2.7
    w2 = 2.0
    w3 = 1.0
    r1 = 1.8
    r2 = -3.4
    r3 = 1.1
    θ1 = 3.72
    θ2 = 5.9
    θ3 = 2.0

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    top1 = classifyCellEigenvalue(M1,M2,M3)

    d1 = 2.3
    d2 = -0.6
    d3 = 1.3
    w1 = 2.7
    w2 = 2.0
    w3 = 1.0
    r1 = -1.1
    r2 = -2.1
    r3 = 0.2
    θ1 = 3.72
    θ2 = 5.9
    θ3 = 2.0

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    top2 = classifyCellEigenvalue(M1,M2,M3)
    println(top1)
    println(top2)
    println(cellTopologyMatches(top1,top2))

end

main()