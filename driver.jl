include("utils.jl")
include("conicUtils.jl")
include("tensorField.jl")
include("huffman.jl")
include("decompress.jl")
include("compress.jl")
include("evaluation.jl")

using .compress
using .decompress
using .tensorField
using .utils

function main()
    folder = "../output/slice"
    dims = (640,80,1)
    eb = 0.01
    edgeError = 1.0
    naive = false
    eigenvalue = true
    eigenvector = true
    minCrossing = 0.0001
    baseCompressor = "sz3"

    compression_start = time()

    if naive
        compress2dNaive(folder, dims, "compressed_output", eb, "../output", baseCompressor)
    else
        compress2d(folder, dims, "compressed_output", eb, edgeError, "../output", true, eigenvalue, eigenvector, minCrossing, baseCompressor)
    end
    compression_end = time()
    ct = compression_end - compression_start

    compressed_size = filesize("../output/compressed_output.tar.zst")
    removeIfExists("../output/compressed_output.tar")
    decompression_start = time()
    if naive
        decompress2dNaive("compressed_output", "reconstructed", "../output", baseCompressor)
    else
        decompress2d("compressed_output", "reconstructed", "../output", baseCompressor)
    end
    decompression_end = time()
    dt = decompression_end - decompression_start

    printEvaluation2d(folder,  "../output/reconstructed", dims, eb, compressed_size, ct, dt, edgeError, eigenvalue, eigenvector, minCrossing)
end

main()