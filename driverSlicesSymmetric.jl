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

    folder = ""
    dims = (-1, -1, -1)
    eb = -1.0
    naive = false
    mask = true
    slice = -1
    bits = 6

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--dataset"
            folder = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--dims"
            dims = (parse(Int64, ARGS[i+1]), parse(Int64, ARGS[i+2]), parse(Int64, ARGS[i+3]))
            i += 4
        elseif ARGS[i] == "--eb"
            eb = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--naive"
            naive = true
            i += 1
        elseif ARGS[i] == "--nomask"
            mask = false
            i += 1
        elseif ARGS[i] == "--slice"
            slice = parse(Int64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--bits"
            bits = parse(Int64, ARGS[i+1])
            i += 2
        else
            println("ERROR: Unknown argument $(ARGS[i])")
            exit(1)
        end
    end

    badArgs = false

    if folder == ""
        println("ERROR: Missing argument --dataset")
        badArgs = true
    end

    if dims == (-1,-1,-1)
        println("ERROR: Missing argument --dims")
        badArgs = true
    end

    if eb == -1.0
        println("ERROR: Missing argument --eb")
        badArgs = true
    end

    if badArgs
        exit(1)
    end

    if slice == -1
        range = 1:dims[3]
    else
        range = slice:slice
    end

    totalCompressionTime = 0.0
    totalDecompressionTime = 0.0
    totalBitrate = 0.0

    tf = loadTensorField2dFromFolder(folder, dims)

    stdout_ = stdout

    for t in range

        println("t = $t")

        redirect_stdout(devnull)

        try
            run(`rm -r ../output/slice`)
        catch
        end

        run(`mkdir ../output/slice`)
        saveArray64("../output/slice/row_1_col_1.dat", tf.entries[1,:,:,t])
        saveArray64("../output/slice/row_1_col_2.dat", tf.entries[2,:,:,t])
        saveArray64("../output/slice/row_2_col_1.dat", tf.entries[3,:,:,t])
        saveArray64("../output/slice/row_2_col_2.dat", tf.entries[4,:,:,t])

        compression_start = time()

        if naive
            if mask
                compress2dSymmetricNaiveWithMask("../output/slice", (dims[1],dims[2],1), "compressed_output", eb)
            else
                compress2dSymmetricNaive("../output/slice", (dims[1],dims[2],1), "compressed_output", eb)
            end
        else
            compress2dSymmetric("../output/slice", (dims[1],dims[2],1), "compressed_output", eb, bits, "../output")
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

        totalCompressionTime += ct
        totalDecompressionTime += dt

        metrics = evaluationList2dSymmetric("../output/slice", "../output/reconstructed", (dims[1], dims[2], 1), compressed_size)

        if !naive && (!metrics[1] || metrics[2] > eb*metrics[3])
            redirect_stdout(stdout_)
            println(metrics[1])
            println(metrics[2])
            println(eb*metrics[3])
            println("failed on slice $t")
            exit(1)
        else
            totalBitrate += metrics[4]
        end

        redirect_stdout(stdout_)

    end

    averageBitrate = totalBitrate
    if slice == -1
        averageBitrate /= dims[3]
    end

    println("compression ratio: $(256.0/averageBitrate)")
    println("compression time: $totalCompressionTime")
    println("decompression time: $totalDecompressionTime")

end

main()