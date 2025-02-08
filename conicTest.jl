include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")
include("cellTopologyOld.jl")
include("tensorField.jl")

using ..cellTopology
using ..cellTopologyOld
using ..tensorField

using StaticArrays

function main()
    # d1 = -1.1
    # d2 = 1.2
    # d3 = 0.5
    # w1 = 0.3
    # w2 = 0.4
    # w3 = 3.0
    # r1 = 0.3
    # r2 = 0.4
    # r3 = 3.0
    # θ1 = 0.0
    # θ2 = 0.0
    # θ3 = 0.1
    d1, r1, w1, θ1 = (0.005188146606087685, 0.02929776906967163, 0.026197967359867128, 3.001343043953188)
    d2, r2, w2, θ2 = (0.001831110566854477, 0.017090365290641785, 0.018723547027061947, 2.8095331860469077)
    d3, r3, w3, θ3 = (0.005188146606087685, 0.03662221133708954, 0.0357066560536623, 3.141592653589793)

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))
    println(M1)
    println(M2)
    println(M3)
    M1 = SMatrix{2,2,Float64}((-0.02075258642435074, -0.025635547935962677, 0.032959990203380585, 0.031128879636526108))
    M2 = SMatrix{2,2,Float64}((-0.0158696249127388, -0.010986663401126862, 0.023194067180156708, 0.019531846046447754))
    M3 = SMatrix{2,2,Float64}((-0.030518509447574615, -0.03662221133708954, 0.03662221133708954, 0.040894802659749985))
    println(decomposeTensor(M1))
    println(decomposeTensor(M2))    
    println(decomposeTensor(M3))
    println("=============")

    top1 = classifyCellEigenvalue(M1,M2,M3,true)
    println(top1)
    top2 = classifyCellEigenvalueOld(M1,M2,M3,true)
    println(top2)
    # top2 = classifyCellEigenvector(M1,M2,M3)
    # println(top2)

end

main()