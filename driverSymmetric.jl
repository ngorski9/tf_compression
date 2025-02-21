include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")
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
    # stress slice 12 (or probably others) makes a good teaser!

    folder = "../output/slice"
    dims = (384,384,1)
    eb = 0.01

    # stress3xy slice 13 0.008899 is equivalent to 0.01 for mine
    # stress3xy slice 24 0.0094
    # stress3xy slice 10 is roughly 0.00875 (sz3)

    # slice 10 (sperr) gives 38.39 0.0089 equiv

    naive =false
    mask = false
    bits = 6 # number of bits used for quantization
    baseCompressor = "sz3"

    println("hi")
    compression_start = time()
    if naive
        if mask
            compress2dSymmetricNaiveWithMask(folder, dims, "compressed_output", eb, "../output", baseCompressor)
        else
            compress2dSymmetricNaive(folder, dims, "compressed_output", eb, "../output", baseCompressor)
        end
    else
        compress2dSymmetric(folder, dims, "compressed_output", eb, bits, "../output", baseCompressor)
    end
    compression_end = time()
    ct = compression_end - compression_start

    compressed_size = filesize("../output/compressed_output.tar.zst")
    removeIfExists("../output/compressed_output.tar")
    decompression_start = time()
    if naive
        if mask
            decompress2dSymmetricNaiveWithMask("compressed_output", "reconstructed", "../output", baseCompressor)
        else
            decompress2dSymmetricNaive("compressed_output", "reconstructed", "../output", baseCompressor)
        end
    else
        decompress2dSymmetric("compressed_output", "reconstructed", bits, "../output", baseCompressor)
    end    
    decompression_end = time()
    dt = decompression_end - decompression_start

    printEvaluation2dSymmetric(folder,  "../output/reconstructed", dims, eb, compressed_size, ct, dt)

end

main()