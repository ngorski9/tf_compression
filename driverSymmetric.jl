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
using .utils

function main()
    # stress slice 12 (or probably others) makes a good teaser!

    folder = "../output/slice"
    dims = (65,65,1)
    eb = 0.009
    naive = true
    mask = true
    bits = 6 # number of bits used for quantization

    symmetric_eval = false

    println("hi")
    compression_start = time()
    if naive
        if mask
            compress2dSymmetricNaiveWithMask(folder, dims, "compressed_output", eb)
        else
            compress2dSymmetricNaive(folder, dims, "compressed_output", eb)    
        end
    else
        compress2dSymmetric(folder, dims, "compressed_output", eb, bits)
    end
    compression_end = time()
    ct = compression_end - compression_start

    compressed_size = filesize("../output/compressed_output.tar.zst")
    removeIfExists("../output/compressed_output.tar")
    decompression_start = time()
    if naive
        if mask
            decompress2dSymmetricNaiveWithMask("compressed_output", "reconstructed")
        else
            decompress2dSymmetricNaive("compressed_output", "reconstructed")
        end
    else
        decompress2dSymmetric("compressed_output", "reconstructed", bits)
    end    
    decompression_end = time()
    dt = decompression_end - decompression_start

    printEvaluation2dSymmetric(folder,  "../output/reconstructed", dims, symmetric_eval, compressed_size, ct, dt)

end

main()