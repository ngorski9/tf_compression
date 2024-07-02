include("utils.jl")
include("tensorField.jl")
include("huffman.jl")
include("decompress.jl")
include("compress.jl")
include("evaluation.jl")
include("plot.jl")

using Plots

using .compress
using .decompress
using .plotTensorField
using .tensorField

folder = "../data/2d/wind1"
dims = (1, 200, 100)
eb = 0.005
naive = false

symmetric_eval = false
display_plots = 0
# 0 = No plots
# 1 = eigenvector & dual eigenvector directions
# 2 = eigenvector graph
# 3 = eigenvalue graph

# Plotting variables
display_frame = 1
display_vector = 1 # 1 for major, 2 for minor
show_borders = false
show_points = false
slice = (1,65,1,65)

println("hi")

compression_start = time()
entropy::Float64 = 0.0
losslessBitrate::Float64 = 0.0

if naive
    compress2dNaive(folder, dims, "compressed_output", eb)
else
    entropy, losslessBitrate = compress2d(folder, dims, "compressed_output", eb)
end
compression_end = time()
ct = compression_end - compression_start

compressed_size = filesize("../output/compressed_output.tar.xz")

decompression_start = time()
if naive
    decompress2dNaive("compressed_output", "reconstructed")
else
    decompress2d("compressed_output", "reconstructed")
end    
decompression_end = time()
dt = decompression_end - decompression_start

printEvaluation2d(folder,  "../output/reconstructed", dims, entropy, losslessBitrate, compressed_size, ct, dt)

if display_plots != 0
    tf1 = loadTensorField2dFromFolder(folder, dims)[1]
    tf2 = loadTensorField2dFromFolder("../output/reconstructed", dims)[1]

    if display_plots == 1
        plot1 = plotEigenFieldGlyphs2d(tf1, display_frame, slice, display_vector)
        plot2 = plotEigenFieldGlyphs2d(tf2, display_frame, slice, display_vector)
    elseif display_plots == 2
        plot1 = plotEigenvectorGraph(tf1, display_frame, slice, show_borders, show_points)
        plot2 = plotEigenvectorGraph(tf2, display_frame, slice, show_borders, show_points)
    elseif display_plots == 3
        plot1 = plotEigenvalueGraph(tf1, display_frame, slice, show_borders, show_points)
        plot2 = plotEigenvalueGraph(tf2, display_frame, slice, show_borders, show_points)
    end

    combined = plot(plot1, plot2, layout=(1,2))
    display(combined)
    println("Press a key to exit")    
    readline()    
end