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

    folder = "../data/asym/heatedCylinder"
    dims = (150,450,2001)
    eb = 0.01
    edgeError = 1.0
    naive = false
    slice = -1

    if slice == -1
        range = 1:dims[3]
    else
        range = slice:slice
    end

    range = 500:2001

    totalCompressionTime = 0.0
    totalDecompressionTime = 0.0
    totalBitrate = 0.0
    maxError = 0.0

    falseVertexEigenvector = 0
    falseVertexEigenvalue = 0
    falseEdge = 0
    falseCell = 0

    tf, dtype = loadTensorField2dFromFolder(folder, dims)

    stdout_ = stdout

    for t in range

        println("t = $t")

        redirect_stdout(devnull)

        try
            run(`rm -r ../output/slice`)
        catch
        end

        run(`mkdir ../output/slice`)
        saveArray("../output/slice/row_1_col_1.dat", tf.entries[1,1][:,:,t])
        saveArray("../output/slice/row_1_col_2.dat", tf.entries[1,2][:,:,t])
        saveArray("../output/slice/row_2_col_1.dat", tf.entries[2,1][:,:,t])
        saveArray("../output/slice/row_2_col_2.dat", tf.entries[2,2][:,:,t])

        compression_start = time()

        if naive
            compress2dNaive("../output/slice", (dims[1],dims[2],1), "compressed_output", eb)
        else
            entropy, losslessBitrate = compress2d("../output/slice", (dims[1],dims[2],1), "compressed_output", eb, edgeError)
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

        totalCompressionTime += ct
        totalDecompressionTime += dt

        metrics = evaluationList2d("../output/slice", "../output/reconstructed", (dims[1], dims[2], 1), compressed_size, edgeError)

        if !naive && (!metrics[1] || metrics[2] > eb*metrics[3])
            redirect_stdout(stdout_)
            println("failed on slice $t")
            exit(1)
        else
            totalBitrate += metrics[4]
        end

        if naive
            falseVertexEigenvector += metrics[5]
            falseVertexEigenvalue += metrics[6]
            falseEdge += metrics[7]
            falseCell += metrics[8]
        end

        redirect_stdout(stdout_)

    end

    averageBitrate = totalBitrate
    if slice == -1
        averageBitrate /= dims[3]
    end

    println("compression ratio: $(256/averageBitrate)")
    println("compression time: $totalCompressionTime")
    println("decompression time: $totalDecompressionTime")

    if naive
        println("false vertex eigenvector: $falseVertexEigenvector")
        println("false vertex eigenvalue: $falseVertexEigenvalue")
        println("false edge: $falseEdge")
        println("false cell: $falseCell")
    end

end

main()