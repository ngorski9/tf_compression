include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")
include("tensorField.jl")
include("huffman.jl")
include("decompress.jl")
include("compress.jl")
include("evaluation.jl")

using .compress
using .decompress
using .tensorField
using .utils

function mainAsym()
    folder = "../output/slice"
    dims = (101,101,1)
    eb = 0.00055
    naive = false
    eigenvalue = false
    eigenvector = true
    baseCompressor = "sz3"
    parameter = 1.0

    driveAsym(folder,dims,eb,naive,eigenvalue,eigenvector,baseCompressor,parameter)
end

function driveAsym(folder,dims,eb,naive,eigenvalue,eigenvector,baseCompressor,parameter)

    # ocean slice 14 (i think its 14) sperr 0.01 27.01, equiv 0.0034 26.98
    # eigenvalue only: 31 equiv 0.0047

    compression_start = time()

    if naive
        compress2dNaive(folder, dims, "compressed_output", eb, "../output", baseCompressor)
    else
        compress2d(folder, dims, "compressed_output", eb, "../output", true, eigenvalue, eigenvector, baseCompressor, parameter)
    end
    compression_end = time()
    ct = compression_end - compression_start

    compressed_size = filesize("../output/compressed_output.tar.zst")
    removeIfExists("../output/compressed_output.tar")
    decompression_start = time()
    if naive
        decompress2dNaive("compressed_output", "reconstructed", "../output", baseCompressor)
    else
        decompress2d("compressed_output", "reconstructed", "../output", baseCompressor, parameter)
    end
    decompression_end = time()
    dt = decompression_end - decompression_start

    printEvaluation2d(folder,  "../output/reconstructed", dims, eb, compressed_size, ct, dt, eigenvalue, eigenvector )
end

#mainAsym()
