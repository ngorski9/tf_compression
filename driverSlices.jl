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

function main()::Cint

    folder = ""
    dims = (-1,-1,-1)
    eb = -1.0
    edgeError = 1.0
    naive = false
    slice = -1
    eigenvalue = false
    eigenvector = false
    minCrossing = 0.0001
    parameter = 1.0
    csv = ""
    output = ""
    sizes = ""
    baseCompressor = "sz3"
    verbose = false

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--dataset"
            folder = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--dims"
            dims = (parse(Int64,ARGS[i+1]), parse(Int64,ARGS[i+2]), parse(Int64,ARGS[i+3]))
            i += 4
        elseif ARGS[i] == "--eb"
            eb = parse(Float64,ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--edgeEB"
            edgeError = parse(Float64,ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--naive"
            naive = true
            i += 1
        elseif ARGS[i] == "--slice"
            slice = parse(Int64,ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--eigenvalue"
            eigenvalue = true
            i += 1
        elseif ARGS[i] == "--eigenvector"
            eigenvector = true
            i += 1
        elseif ARGS[i] == "--both"
            eigenvalue = true
            eigenvector = true
            i += 1
        elseif ARGS[i] == "--csv"
            csv = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--output"
            output = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--sizes"
            sizes = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--baseCompressor"
            baseCompressor = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--minCrossing"
            minCrossing = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--compressorParameter"
            parameter = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--verbose"
            verbose = true
            i += 1
        else
            println("ERROR: unknown argument $(ARGS[i])")
            exit(1)
        end
    end

    badArgs = false

    if folder == ""
        println("ERROR: missing argument --dataset")
        badArgs = true
    end

    if dims == (-1,-1,-1)
        println("ERROR: missing argument --dims")
        badArgs = true
    end

    if eb == -1.0
        println("ERROR: missing argument --eb")
        badArgs = true
    end

    if output == ""
        println("ERROR: missing argument --output")
        badArgs = true
    end

    if baseCompressor == ""
        println("ERROR: missing argument --baseCompressor")
        badArgs = true
    end

    if !eigenvalue && !eigenvector
        println("ERROR: no preservation specified. Use --eigenvalue, --eigenvector, or --both")
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
    totalMSEByRangeSquared = 0.0
    totalFrobeniusMSEByRangeSquared = 0.0
    maxErrorByRange = 0.0
    totalVertexMatching = [0.0 0.0 ; 0.0 0.0]
    totalEdgeMatching = [0.0 0.0 ; 0.0 0.0]
    totalCellMatching = [0,0,0,0,0,0,0]
    totalCellTypeFrequenciesGround = [0,0,0,0]
    totalCellTypeFrequenciesRecon = [0,0,0,0]
    totalEllipseMatching = [0,0,0,0,0,0,0,0]
    ctv = zeros(Float64, (15,)) # compression time vector
    dtv = zeros(Float64, (8,)) # decompression time vector
    
    falseVertexEigenvector = 0
    falseVertexEigenvalue = 0
    falseEdgeEigenvalue = 0
    falseEdgeEigenvector = 0
    falseCell = 0
    falseEllipse = 0

    tf = loadTensorField2dFromFolder(folder, dims)

    stdout_ = stdout

    trialStart = time()

    for t in range

        println("t = $t")

        if !verbose
            redirect_stdout(devnull)
        end

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
            compress2dNaive("$output/slice", (dims[1],dims[2],1), "compressed_output", eb, output, baseCompressor)
        else
            compressionList = compress2d("$output/slice", (dims[1],dims[2],1), "compressed_output", eb, edgeError, output, false, eigenvalue, eigenvector, minCrossing, baseCompressor, parameter)
            ctv += compressionList
        end

        compression_end = time()
        ct = compression_end - compression_start

        compressed_size = filesize("$output/compressed_output.tar.zst")

        if sizes != ""
            sizes_file = open(sizes, "a")
            write(sizes_file, string(compressed_size) * "\n")
        end

        removeIfExists("$output/compressed_output.tar")
        decompression_start = time()
        if naive
            decompress2dNaive("compressed_output", "reconstructed", output, baseCompressor)
        else
            decompressionList = decompress2d("compressed_output", "reconstructed", output, baseCompressor, parameter)
            dtv += decompressionList
        end
        decompression_end = time()
        dt = decompression_end - decompression_start

        totalCompressionTime += ct
        totalDecompressionTime += dt

        metrics = evaluationList2d("$output/slice", "$output/reconstructed", (dims[1], dims[2], 1), eb, compressed_size, edgeError, eigenvalue, eigenvector, minCrossing)

        if !naive && !metrics[1]
            redirect_stdout(stdout_)
            println("failed on slice $t")
            exit(1)
        end

        totalBitrate += metrics[2]
        maxErrorByRange = max(maxErrorByRange, metrics[3])
        totalMSEByRangeSquared += metrics[4]
        totalFrobeniusMSEByRangeSquared += metrics[5]
        totalVertexMatching += metrics[6]
        totalEdgeMatching += metrics[7]
        totalCellMatching += metrics[8]
        totalCellTypeFrequenciesGround += metrics[9]
        totalCellTypeFrequenciesRecon += metrics[10]
        totalEllipseMatching += metrics[11]

        if naive
            falseVertexEigenvalue += metrics[6][1,2]
            falseVertexEigenvector += metrics[6][2,2]
            falseEdgeEigenvalue += metrics[7][1,2]
            falseEdgeEigenvector += metrics[7][2,2]
            falseCell += sum(metrics[8]) - metrics[8][1]
            falseEllipse += metrics[11][5] + metrics[11][6]
            if eigenvalue
                falseEllipse += metrics[11][2] + metrics[11][3]
            end
        end

        redirect_stdout(stdout_)

    end

    trialEnd = time()
    trialTime = trialEnd - trialStart

    averageBitrate = totalBitrate
    if slice == -1
        averageBitrate /= dims[3]
    end
    ratio = 256.0/averageBitrate

    averageMSEByRangeSquared = totalMSEByRangeSquared
    averageFrobeniusMSEByRangeSquared = totalFrobeniusMSEByRangeSquared

    if slice == -1
        averageMSEByRangeSquared /= dims[3]
        averageFrobeniusMSEByRangeSquared /= dims[3]
    end

    psnr = -10 * log(10, averageMSEByRangeSquared)

    println("compression ratio: $ratio")
    println("psnr: $psnr")
    println("compression time: $totalCompressionTime")
    println("decompression time: $totalDecompressionTime")
    println("trial time: $trialTime")

    if naive
        println("false vertex eigenvalue: $falseVertexEigenvalue")
        println("false vertex eigenvector: $falseVertexEigenvector")
        println("false edge eigenvalue: $falseEdgeEigenvalue")
        println("false edge eigenvector: $falseEdgeEigenvector")
        println("false cell: $falseCell")
        println("false ellipse: $falseEllipse")
    end

    if csv != ""
        if isfile(csv)
            outf = open(csv, "a")
        else
            outf = open(csv, "w")

            # write header :(
            write(outf, "dataset,target,eb,edgeEB,ratio,max error,psnr,ct,dt,tt,mse,frobeniusMse,points,edges,circular points,cp types (ground),cp types (recon),")
            write(outf, "false ellipses,numPoints,numCells,setup 1,bc,setup 2,proc. points,ellipse check,proc. edges,")
            write(outf, "proc. cp,proc. ellipse,queue, total proc.,num corrected,num proc.'d,write comp.,lossless comp.,comp. clean,decomp. zstd,")
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
        if eigenvector
            if eigenvalue
                target = "BOTH"
            else
                target = "EIGENVECTOR"
            end
        else
            target = "EIGENVALUE"
        end

        if naive
            target = target * " (NAIVE)"
        end

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

        totalVertexMatchingS = s(totalVertexMatching)
        totalEdgeMatchingS = s(totalEdgeMatching)
        totalCellMatchingS = s(totalCellMatching)
        totalCellTypeFrequenciesGroundS = s(totalCellTypeFrequenciesGround)
        totalCellTypeFrequenciesReconS = s(totalCellTypeFrequenciesRecon)
        totalEllipseMatchingS = s(totalEllipseMatching)

        # write the data :((
        write(outf, "$name,$target,$eb,$edgeError,$ratio,$maxErrorByRange,$psnr,$totalCompressionTime,$totalDecompressionTime,$trialTime,$averageMSEByRangeSquared,")
        write(outf, "$averageFrobeniusMSEByRangeSquared,$totalVertexMatchingS,$totalEdgeMatchingS,$totalCellMatchingS,$totalCellTypeFrequenciesGroundS,")
        write(outf, "$totalCellTypeFrequenciesReconS,$totalEllipseMatchingS,$numPoints,$numCells,$(ctv[1]),$(ctv[2]),$(ctv[3]),$(ctv[4]),$(ctv[5]),$(ctv[6]),")
        write(outf, "$(ctv[7]),$(ctv[8]),$(ctv[9]),$(ctv[10]),$(ctv[11]),$(ctv[12]),$(ctv[13]),$(ctv[14]),$(ctv[15]),$(dtv[1]),$(dtv[2]),$(dtv[3]),$(dtv[4]),")
        write(outf, "$(dtv[5]),$(dtv[6]),$(dtv[7]),$(dtv[8])\n")

    end

    return 0

end

main()
