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

    decomp1 = (-0.006103701889514923, 0.0027466658502817154, 0.005329828541004099, -0.4124104415973873)
    decomp2 = (-0.006714072078466415, -0.0024414807558059692, 0.0025166207597752126, -1.8157749899217608)
    decomp3 = (-0.008239997550845146, 0.00030518509447574615, 0.002763568568896735, -0.11065722117389563)

    d1, r1, w1, θ1 = decomp1
    d2, r2, w2, θ2 = decomp2
    d3, r3, w3, θ3 = decomp3

    M1 = SMatrix{2,2,Float64}(( d1 + w1*cos(θ1), r1 + w1*sin(θ1), -r1+w1*sin(θ1), d1-w1*cos(θ1) ))
    M2 = SMatrix{2,2,Float64}(( d2 + w2*cos(θ2), r2 + w2*sin(θ2), -r2+w2*sin(θ2), d2-w2*cos(θ2) ))
    M3 = SMatrix{2,2,Float64}(( d3 + w3*cos(θ3), r3 + w3*sin(θ3), -r3+w3*sin(θ3), d3-w3*cos(θ3) ))

    # M1 = SMatrix{2,2,Float64}((-0.00011995717358732921, -6.502135191598569e-6, -8.897467989487296e-6,  0.00012480001277655222))
    # M2 = SMatrix{2,2,Float64}((1.8782440970925977e-6,  -1.4496063672295506e-7, 9.794174361859275e-8, 2.610598835822252e-6))
    # M3 = SMatrix{2,2,Float64}((2.141215730527772e-5, -3.6564157537227297e-7, -2.8467736449483924e-6, -1.639844407047501e-5))

    top1 = classifyCellEigenvalue(M1,M2,M3,true)
    println(top1)

    top2 = classifyCellEigenvalue(M1,M2,M3,false)
    println(top2)

end

main()