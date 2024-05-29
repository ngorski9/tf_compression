include("utils.jl")
include("tensorField.jl")
include("plot.jl")

using Plots

using .tensorField
using .plotTensorField
using .utils

a = [-0.006904032547026873 0.01214652694761753; 0.008542274124920368 -0.009403611533343792]
b = [-0.0018496820703148842 -0.0067755491472780704; 0.02071954496204853 -0.023761611431837082]

println(classifyEdgeEigenvalue(a,b))
println(classifyEdgeEigenvalue(b,a))

# file = "../data/2d/asym1"
# dims = (25,65,65)

# x = time()
# tf = loadTensorField2dFromFolder(file, dims)[1]

# r = zeros(Float32, dims)
# a = zeros(Float32, dims)
# b = zeros(Float32, dims)
# θ = zeros(Float32, dims)

# for j in 1:dims[3]
#     for i in 1:dims[2]
#         for t in 1:dims[1]
#             tensor = getTensor(tf, t, i, j)
#             yd, yr, ys, θ_ = decomposeTensor(tensor)
#             r_ = sqrt(yd^2+yr^2+ys^2)
#             a_ = sign(yd)*(yd^2)/(r_^2)
#             b_ = sign(yr)*(yr^2)/(r_^2)

#             r[t,i,j] = r_
#             a[t,i,j] = a_
#             b[t,i,j] = b_
#             θ[t,i,j] = θ_
#         end
#     end
# end

# saveArray("../output/r.dat", r)
# saveArray("../output/a.dat", a)
# saveArray("../output/b.dat", b)
# saveArray("../output/theta.dat", θ)

# output = "../output"
# θ_bound = pi/180

# run(`../SZ3-master/build/bin/sz3 -f -i $output/theta.dat -z $output/theta.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $θ_bound`)
# run(`../SZ3-master/build/bin/sz3 -f -i $output/a.dat -z $output/a.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS 0.02`)
# run(`../SZ3-master/build/bin/sz3 -f -i $output/b.dat -z $output/b.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS 0.02`)
# run(`../SZ3-master/build/bin/sz3 -f -i $output/r.dat -z $output/r.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M REL 0.02`)

# exit()

# #a = plotEigenFieldGlyphs2d(tf, 14, 1)
# a = plotEigenvalueGraph(tf, 14, false, false)
# #a = plotEigenvectorGraph(tf, 14)
# display(a)
# println(time()-x)

# readline()









# include("huffman.jl")
# include("tensor.jl")
# include("plot.jl")
# include("utils.jl")


# function make_tensor(x,y)
#     return [ (x-12) (y-12) ; (x-12) (y-12) ]
# end

# size = 25

# topLeft = zeros(Float64, (1, size, size))
# topRight = zeros(Float64, (1, size, size))
# bottomLeft = zeros(Float64, (1, size, size))
# bottomRight = zeros(Float64, (1, size, size))

# for i in 1:size
#     for j in 1:size
#         tensor = make_tensor(i,j)
#         topLeft[1, i, j] = tensor[1,1]
#         topRight[1, i, j] = tensor[1,2]
#         bottomLeft[1, i, j] = tensor[2,1]
#         bottomRight[1, i, j] = tensor[2,2]
#     end
# end

# save_array("../output/test/row_1_col_1.dat", topLeft)
# save_array("../output/test/row_1_col_2.dat", topRight)
# save_array("../output/test/row_2_col_1.dat", bottomLeft)
# save_array("../output/test/row_2_col_2.dat", bottomRight)

# tf = loadSymmetricTensorField2dFromFolder("../output/test", (1,size,size))[1]

# q = plotEigenFieldGlyphs2d(tf, 1, 1)
# display(q)

# readline()