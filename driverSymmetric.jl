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
    folder = "../data/sym/brain23"
    dims = (66,108,108)
    eb = 0.01
    naive = false
    bits = 6 # number of bits used for quantization

    symmetric_eval = false

    println("hi")
    compression_start = time()
    if naive
        compress2dSymmetricNaive(folder, dims, "compressed_output", eb)    
    else
        compress2dSymmetric(folder, dims, "compressed_output", eb, bits)
    end
    compression_end = time()
    ct = compression_end - compression_start

    compressed_size = filesize("../output/compressed_output.tar.xz")

    decompression_start = time()
    if naive
        decompress2dSymmetricNaive("compressed_output", "reconstructed")
    else
        decompress2dSymmetric("compressed_output", "reconstructed", bits)
    end    
    decompression_end = time()
    dt = decompression_end - decompression_start

    printEvaluation2dSymmetric(folder,  "../output/reconstructed", dims, symmetric_eval, compressed_size, ct, dt)

end

main()