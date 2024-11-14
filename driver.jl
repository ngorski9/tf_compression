include("utils.jl")
include("tensorField.jl")
include("huffman.jl")
include("decompress.jl")
include("compress.jl")
include("evaluation.jl")

using Plots

using .compress
using .decompress
using .tensorField

function main()
    folder = "../output/slice"
    dims = (150,450,1)
    eb = 0.05
    edgeError = 0.2
    naive = false

    compression_start = time()
    entropy::Float64 = 0.0
    losslessBitrate::Float64 = 0.0

    if naive
        compress2dNaive(folder, dims, "compressed_output", eb)
    else
        entropy, losslessBitrate = compress2d(folder, dims, "compressed_output", eb, edgeError)
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

    printEvaluation2d(folder,  "../output/reconstructed", dims, entropy, losslessBitrate, compressed_size, ct, dt, edgeError)
end

main()