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

function main()

    folder = ""
    dims = (-1, -1, -1)
    eb = -1.0
    naive = false
    mask = true
    slice = -1
    bits = 6
    csv = ""
    output = ""
    baseCompressor = "sz3"
    skipEval = false

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
        elseif ARGS[i] == "--csv"
            csv = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--output"
            output = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--baseCompressor"
            baseCompressor = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--skipEval"
            skipEval = true
            i += 1
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

    if output == ""
        println("ERROR: Missing argument --output")
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
    maxErrorByRange = 0.0
    ctv = zeros(Float64, (12,))
    dtv = zeros(Float64, (8,))
    totalMSEByRangeSquared = 0.0
    totalFrobeniusMSEByRangeSquared = 0.0
    totalCellMatching = [0,0,0,0,0,0,0]
    totalCellTypeFrequenciesGround = [0,0,0,0,0,0]
    totalCellTypeFrequenciesRecon = [0,0,0,0,0,0]
    numNonzeroSlices = 0

    tf = loadTensorField2dFromFolder(folder, dims)

    stdout_ = stdout

    trialStart = time()
    for t in range

        println("t = $t")

        redirect_stdout(devnull)

        try
            run(`rm -r $output`)
            run(`mkdir $output`)
        catch
        end

        run(`mkdir $output/slice`)
        saveArray64("$output/slice/row_1_col_1.dat", tf.entries[1,:,:,t])
        saveArray64("$output/slice/row_1_col_2.dat", tf.entries[2,:,:,t])
        saveArray64("$output/slice/row_2_col_1.dat", tf.entries[3,:,:,t])
        saveArray64("$output/slice/row_2_col_2.dat", tf.entries[4,:,:,t])

        compression_start = time()

        if naive
            if mask
                compress2dSymmetricNaiveWithMask("$output/slice", (dims[1],dims[2],1), "compressed_output", eb, output, baseCompressor)
            else
                compress2dSymmetricNaive("$output/slice", (dims[1],dims[2],1), "compressed_output", eb, output, baseCompressor)
            end
        else
            ctVector = compress2dSymmetric("$output/slice", (dims[1],dims[2],1), "compressed_output", eb, bits, output, baseCompressor)
            ctv += ctVector
        end

        compression_end = time()
        ct = compression_end - compression_start

        compressed_size = filesize("$output/compressed_output.tar.zst")
        removeIfExists("$output/compressed_output.tar")
        decompression_start = time()
        if naive
            if mask
                decompress2dSymmetricNaiveWithMask("compressed_output", "reconstructed", output, baseCompressor)
            else
                decompress2dSymmetricNaive("compressed_output", "reconstructed", output, baseCompressor)
            end
        else
            dtVector = decompress2dSymmetric("compressed_output", "reconstructed", bits, output, baseCompressor)
            dtv += dtVector
        end
        decompression_end = time()
        dt = decompression_end - decompression_start

        totalCompressionTime += ct
        totalDecompressionTime += dt
        redirect_stdout(stdout_)

        if !skipEval
            metrics = evaluationList2dSymmetric("$output/slice", "$output/reconstructed", (dims[1], dims[2], 1), eb, compressed_size)

            if !naive && !metrics[1]
                redirect_stdout(stdout_)
                println("failed on slice $t")
                exit(1)
            end

            totalBitrate += metrics[2]
            # a negative value means that the slice is all zeros
            if metrics[3] >= 0
                maxErrorByRange = max(maxErrorByRange, metrics[3])
                totalMSEByRangeSquared += metrics[4]
                totalFrobeniusMSEByRangeSquared += metrics[5]
                numNonzeroSlices += 1
            end

            totalCellMatching += metrics[6]
            totalCellTypeFrequenciesGround += metrics[7]
            totalCellTypeFrequenciesRecon += metrics[8]
        else
            totalBitrate += compressed_size*8/(dims[1]*dims[2])
        end

        redirect_stdout(stdout_)

    end
    trialEnd = time()
    trialTime = trialEnd - trialStart

    averageBitrate = totalBitrate
    if slice == -1
        averageBitrate /= dims[3]
    end

    ratio = 192.0/averageBitrate

    println("compression ratio: $ratio")
    println("compression time: $totalCompressionTime")
    println("decompression time: $totalDecompressionTime")
    println("trial time: $trialTime")

    if csv != ""
        if isfile(csv)
            outf = open(csv, "a")
        else
            outf = open(csv, "w")

            # write header :(
                write(outf, "dataset,target,bc,eb,ratio,max error,psnr,ct,dt,tt,mse,frobeniusMse,fp val,fp vec,ft val,ft vec,cells,cp types (ground),cp types (recon),")
                write(outf, "numPoints,numCells,setup 1,bc,setup 2,proc. points,")
                write(outf, "proc. cp,proc. cells,queue,total proc.,num corrected,num proc.'d,write comp.,lossless comp.,comp. clean,decomp. zstd,")
                write(outf, "tar,load,base decomp.,read base decomp.,augment,save decomp.,cleanup\n")
        end

        # compute composite data

        # compute the name of the dataset
        if last(folder) == '/'
            folder = folder[1:lastindex(folder)-1]
        end

        if occursin("/",folder)
            name = folder[first(findlast("/",folder))+1:lastindex(folder)]
        else
            name = folder
        end

        if slice != -1
            name = name * " (slice $slice)"
        end

        # compute the target
        target = "SYM"
        if !mask
            target = target * "/NOMASK"
        end

        if naive
            target = target * " (NAIVE)"
        elseif bits != 6
            target = target * " ($bits BITS)"
        end

        averageMSEByRangeSquared = totalMSEByRangeSquared / numNonzeroSlices
        averageFrobeniusMSEByRangeSquared = totalFrobeniusMSEByRangeSquared / numNonzeroSlices

        psnr = -10 * log(10, averageMSEByRangeSquared)

        numPoints = dims[1]*dims[2]
        numCells = (dims[1]-1)*(dims[2]-1)*2
        if slice == -1
            numPoints *= dims[3]
            numCells *= dims[3]
        end

        function s(l)
            str = string(l)
            str = replace(str, "," => "")
            str = replace(str, ";" => "")
            return str
        end

        totalCellMatchingS = s(totalCellMatching)
        totalCellTypeFrequenciesGroundS = s(totalCellTypeFrequenciesGround)
        totalCellTypeFrequenciesReconS = s(totalCellTypeFrequenciesRecon)

        # write the data :((
        write(outf, "$name,$target,$baseCompressor,$eb,$ratio,$maxErrorByRange,$psnr,$totalCompressionTime,$totalDecompressionTime,$trialTime,")
        write(outf, "$averageMSEByRangeSquared,$averageFrobeniusMSEByRangeSquared,,,,,$totalCellMatchingS,$totalCellTypeFrequenciesGroundS, $totalCellTypeFrequenciesReconS,")
        write(outf, "$numPoints,$numCells,$(ctv[1]),$(ctv[2]),$(ctv[3]),$(ctv[4]),$(ctv[5]),,$(ctv[6]),$(ctv[7]),")
        write(outf, "$(ctv[8]),$(ctv[9]),$(ctv[10]),$(ctv[11]),$(ctv[12]),$(dtv[1]),$(dtv[2]),$(dtv[3]),$(dtv[4]),")
        write(outf, "$(dtv[5]),$(dtv[6]),$(dtv[7]),$(dtv[8])\n")

    end

end

main()