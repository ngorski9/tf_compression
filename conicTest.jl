include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")

using ..cellTopology

using StaticArrays

function main()
    d1 = -1.1
    d2 = 1.2
    d3 = 0.5
    w1 = 0.3
    w2 = 0.4
    w3 = 3.0
    r1 = 0.3
    r2 = 0.4
    r3 = 3.0
    θ1 = 0.0
    θ2 = 0.0
    θ3 = 0.1

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    top1 = classifyCellEigenvalue(M1,M2,M3,true)
    println(top1)
    # top2 = classifyCellEigenvector(M1,M2,M3)
    # println(top2)

end

main()