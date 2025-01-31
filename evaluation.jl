using Statistics

using ..tensorField
using ..conicUtils
using ..cellTopology

function getCPTypeFrequencies(tf::TensorField2d)
    x,y,T = tf.dims
    types = [0,0,0,0]
    for t in 1:T
        for i in 1:(x-1)
            for j in 1:(y-1)
                for k in 0:1
                    type = getCircularPointType(tf, i, j, t, Bool(k))
                    types[type+1] += 1
                end
            end
        end
    end
    return types
end

function getCPTypeFrequencies(tf::TensorField2dSymmetric)
    x,y,T = tf.dims
    types = [0,0,0,0]
    for t in 1:T
        for i in 1:(x-1)
            for j in 1:(y-1)
                for k in 0:1
                    type = getCriticalType(tf, i, j, t, Bool(k))
                    types[type+1] += 1
                end
            end
        end
    end
    return types
end

function topologyVertexMatching(tf1::TensorField2d, tf2::TensorField2d)

    result = [0 0 ; 0 0]

    # row 1: eigenvector
    # row 2: eigenvalue
    # col 1: match
    # col 2: miss

    x,y,T = tf1.dims
    for t in 1:T
        for j in 1:y
            for i in 1:x

                t1 = getTensor(tf1, i, j, t)
                t2 = getTensor(tf2, i, j, t)

                d1, r1, s1, _ = decomposeTensor(t1)
                d2, r2, s2, _ = decomposeTensor(t2)

                if classifyTensorEigenvalue(d1, r1, s1) == classifyTensorEigenvalue(d2, r2, s2)
                    result[1,1] += 1
                else
                    result[1,2] += 1
                end

                if classifyTensorEigenvector(r1, s1) == classifyTensorEigenvector(r2, s2)
                    result[2,1] += 1
                else
                    result[2,2] += 1
                end

            end
        end
    end

    return result
end

function topologyCellMatching(tf1::TensorField2d, tf2::TensorField2d)

    # Same, FP, FN, FT, FP (degenerate), FN (degenerate), FT (degenerate)
    # Critical -> Degen or Degen -> Critical = FT (degenerate)
    result = [0,0,0,0,0,0,0,0,0]

    SAME = 1 # checks circular points
    FP = 2
    FN = 3
    FT = 4
    FPD = 5
    FND = 6
    FTD = 7
    TSAME = 8 # does the internal topology match or no
    TDIF = 9

    x, y, T = tf1.dims
    x -= 1
    y -= 1
    for t in 1:T
        for j in 1:y
            for i in 1:x                
                for k in 0:1

                    c1 = getCircularPointType(tf1, i, j, t, Bool(k))
                    c2 = getCircularPointType(tf2, i, j, t, Bool(k))

                    if c1 == c2
                        result[SAME] += 1
                    elseif c1 == CP_NORMAL
                        if c2 == CP_TRISECTOR || c2 == CP_WEDGE
                            result[FP] += 1
                        else
                            result[FPD] += 1
                        end
                    elseif c1 == CP_TRISECTOR || c1 == CP_WEDGE
                        if c2 == CP_NORMAL
                            result[FN] += 1
                        elseif c2 == CP_TRISECTOR || c2 == CP_WEDGE
                            result[FT] += 1
                        else
                            result[FTD] += 1
                        end
                    elseif c1 == CP_ERROR
                        if c2 == CP_NORMAL
                            result[FND] += 1
                        else
                            result[FTD] += 1
                        end
                    end

                    

                end
            end
        end
    end

    return result

end

function topologyCellMatching(tf1::TensorField2dSymmetric, tf2::TensorField2dSymmetric)
    # Same, FP, FN, FT, FP (degenerate), FN (degenerate), FT (degenerate)
    # Critical -> Degen or Degen -> Critical = FT (degenerate)
    result = [0,0,0,0,0,0,0]

    SAME = 1
    FP = 2
    FN = 3
    FT = 4
    FPD = 5
    FND = 6
    FTD = 7

    x, y, T = tf1.dims
    x -= 1
    y -= 1
    for t in 1:T
        for j in 1:y
            for i in 1:x                
                for k in 0:1

                    c1 = getCriticalType(tf1, i, j, t, Bool(k))
                    c2 = getCriticalType(tf2, i, j, t, Bool(k))

                    if c1 == c2
                        result[SAME] += 1
                    elseif c1 == CP_NORMAL
                        if c2 == CP_TRISECTOR || c2 == CP_WEDGE
                            result[FP] += 1
                        else
                            result[FPD] += 1
                        end
                    elseif c1 == CP_TRISECTOR || c1 == CP_WEDGE
                        if c2 == CP_NORMAL
                            result[FN] += 1
                        elseif c2 == CP_TRISECTOR || c2 == CP_WEDGE
                            result[FT] += 1
                        else
                            result[FTD] += 1
                        end
                    elseif c1 == CP_ERROR
                        if c2 == CP_NORMAL
                            result[FND] += 1
                        else
                            result[FTD] += 1
                        end
                    end

                end
            end
        end
    end

    return result
end

# Does not check whether ellipses actually match in the eigenvalue / eigenvector graph
# (as actually detecting if an elliptical boundary shows up is extremely difficult)
# but just detects whether ellipses are contained within the cell boundaries.
function ellipseMatching(tf1::TensorField2d, tf2::TensorField2d)

    results = [0,0,0,0,0,0,0,0]
    matchDS = 1
    fpDS = 2
    fnDS = 3
    matchRS = 4
    fpRS = 5
    fnRS = 6
    numDS = 7
    numRS = 8

    x, y, T = tf1.dims
    x -= 1
    y -= 1

    for t in 1:T
        for j in 1:y
            for i in 1:x                
                for k in 0:1

                    tensors1 = getTensorsAtCell(tf1,i,j,t,Bool(k))
                    tensors2 = getTensorsAtCell(tf2,i,j,t,Bool(k))

                    rs1 = checkRSEllipseProper(tensors1...)
                    rs2 = checkRSEllipseProper(tensors2...)

                    if rs1 == rs2
                        results[matchRS] += 1
                    elseif rs1 && !rs2
                        results[fnRS] += 1
                    else
                        results[fpRS] += 1
                    end

                    if rs1
                        results[numRS] += 1
                    end

                    ds1 = checkDSEllipseProper(tensors1...)
                    ds2 = checkDSEllipseProper(tensors2...)

                    if ds1 == ds2
                        results[matchDS] += 1
                    elseif ds1 && !ds2
                        results[fnDS] += 1
                    else
                        results[fpDS] += 1
                    end
                    
                    if ds1
                        results[numDS] += 1
                    end

                end
            end
        end
    end

    return results
end

function reconstructionQuality(tf_ground::TensorField2d, tf_reconstructed::TensorField2d)
    min_val, max_val = getMinAndMax(tf_ground)
    
    if min_val == max_val
        return -1.0, -1.0, -1.0
    end

    peak_signal = max_val - min_val
    mse = mean( (tf_ground.entries - tf_reconstructed.entries) .^ 2 )
    psnr = 10 * log(10, peak_signal^2 / mse)
    frobeniusMse = mean( sum( tf_ground.entries - tf_reconstructed.entries, dims=1 ) .^ 2 )
    return psnr, mse, frobeniusMse
end

function reconstructionQuality(tf_ground::TensorField2dSymmetric, tf_reconstructed::TensorField2dSymmetric)
    min_val, max_val = getMinAndMax(tf_ground)

    if min_val == max_val
        return -1.0, -1.0, -1.0
    end

    peak_signal = max_val - min_val
    mse = mean( (tf_ground.entries - tf_reconstructed.entries) .^ 2 )
    psnr = 10 * log(10, peak_signal^2 / mse)
    frobeniusMse = mean( sum( tf_ground.entries - tf_reconstructed.entries, dims=1 ) .^ 2 )
    return psnr, mse, frobeniusMse
end

function maxErrorAndRange(tf_ground::TensorField2d, tf_reconstructed::TensorField2d)

    max_val = maximum(tf_ground.entries)
    min_val = minimum(tf_ground.entries)
    max_error = maximum( abs.(tf_ground.entries - tf_reconstructed.entries) )

    return (max_error), (max_val - min_val)
end

function maxErrorAndRange(tf_ground::TensorField2dSymmetric, tf_reconstructed::TensorField2dSymmetric)

    max_val = maximum(tf_ground.entries)
    min_val = minimum(tf_ground.entries)
    max_error = maximum( abs.(tf_ground.entries - tf_reconstructed.entries) )

    return (max_error), (max_val - min_val)

end

function printEvaluation2d(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, eb::Float64, compressed_size::Int64 = -1, compression_time::Float64 = -1.0, decompression_time::Float64 = -1.0, edgeEB = 1.0, eigenvalue = true, eigenvector = true, minCrossing = 0.01)
    tf1 = loadTensorField2dFromFolder(ground, dims)
    tf2 = loadTensorField2dFromFolder(reconstructed, dims)

    vertexMatching = topologyVertexMatching(tf1, tf2)
    edgeMatching = topologyEdgeMatching(tf1, tf2, edgeEB, minCrossing)
    cellMatching = topologyCellMatching(tf1, tf2)
    ellipseMatching_ = ellipseMatching(tf1, tf2)

    println("type frequencies: $(getTypeFrequencies(tf1))")
    println("type frequencies: $(getTypeFrequencies(tf2))")

    if (!eigenvector || vertexMatching[2,2] == 0) && (!eigenvalue || vertexMatching[1,2] == 0) && (!eigenvalue || edgeMatching[1,2] == 0) && (!eigenvector || edgeMatching[2,2] == 0) && (!eigenvector || sum(cellMatching) == cellMatching[1]) && (!eigenvalue || (ellipseMatching_[2] == 0 && ellipseMatching_[3] == 0)) && (ellipseMatching_[5] == 0 && ellipseMatching_[6] == 0)
        result = "GOOD"
    else
        result = "BAD"
    end

    psnr, _, _ = reconstructionQuality(tf1, tf2)

    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])

    eltSize = 64

    ratio = eltSize*4/bitrate

    max_error, range = maxErrorAndRange(tf1, tf2)

    if eb*range < max_error
        result = "BAD"
    end

    max_error /= range

    println("-----------------")
    println(result)
    println("\tvertex eigenvalue: ($(vertexMatching[1,1]), $(vertexMatching[1,2]))")    
    println("\tvertex eigenvector: ($(vertexMatching[2,1]),$(vertexMatching[2,2]))")
    println("\tedge eigenvalue: ($(edgeMatching[1,1]), $(edgeMatching[1,2]))")
    println("\tedge eigenvector: ($(edgeMatching[2,1]), $(edgeMatching[2,2]))")    
    println("\tcell matching: ($(cellMatching[1]), $(sum(cellMatching)-cellMatching[1]))")
    println("\tellipse matching: $ellipseMatching_")

    println("\nmax error: $max_error")
    println("compression ratio: $ratio")
    println("PSNR: $psnr")
    if compression_time != -1
        println("compression time: $compression_time")
    end
    if decompression_time != -1
        println("decompression time: $decompression_time")
    end

end

function evaluationList2d(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, eb::Float64, compressed_size::Int64 = -1, edgeEB = 1.0, eigenvalue = true, eigenvector = true, minCrossing = 0.01)
    tf1 = loadTensorField2dFromFolder(ground, dims)
    tf2 = loadTensorField2dFromFolder(reconstructed, dims)

    vertexMatching = topologyVertexMatching(tf1, tf2)
    edgeMatching = topologyEdgeMatching(tf1, tf2, edgeEB, minCrossing)
    cellMatching = topologyCellMatching(tf1, tf2)
    cellTypeFrequenciesGround = getTypeFrequencies(tf1)
    cellTypeFrequenciesRecon = getTypeFrequencies(tf2)
    ellipseMatching_ = ellipseMatching(tf1, tf2)
    _, mse, frobeniusmse = reconstructionQuality(tf1, tf2)

    if (!eigenvector || vertexMatching[2,2] == 0) && (!eigenvalue || vertexMatching[1,2] == 0) && (!eigenvalue || edgeMatching[1,2] == 0) && (!eigenvector || edgeMatching[2,2] == 0) && (!eigenvector || sum(cellMatching) == cellMatching[1]) && (!eigenvalue || (ellipseMatching_[2] == 0 && ellipseMatching_[3] == 0)) && (ellipseMatching_[5] == 0 && ellipseMatching_[6] == 0)
        preserved = true
    else
        preserved = false
    end

    # psnr = asymmetricPSNR(tf1, tf2) # going to ignore this for now...
    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])
    max_error, range = maxErrorAndRange(tf1, tf2)

    if max_error > eb * range
        preserved = false
    end

    if range != 0.0
        mseByRangeSquared = mse / range^2
        frobeniusMseByRangeSquared = frobeniusmse / range^2
        maxErrorByRange = max_error / range
    else
        mseByRangeSquared = -1.0
        frobeniusMseByRangeSquared = -1.0
        maxErrorByRange = -1.0
    end

    return (preserved, bitrate, maxErrorByRange, mseByRangeSquared, frobeniusMseByRangeSquared, vertexMatching, edgeMatching, cellMatching, cellTypeFrequenciesGround, cellTypeFrequenciesRecon, ellipseMatching_)

end

function printEvaluation2dSymmetric(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, eb::Float64, compressed_size::Int64 = -1, compression_time::Float64 = -1.0, decompression_time::Float64 = -1.0 )
    tf1 = loadTensorField2dSymmetricFromFolder(ground, dims)
    tf2 = loadTensorField2dSymmetricFromFolder(reconstructed, dims)

    psnr, _, _ = reconstructionQuality(tf1, tf2)

    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])

    eltSize = 64

    ratio = eltSize*3/bitrate

    max_error, range = maxErrorAndRange(tf1, tf2)

    println("-----------------")
    cellMatching = topologyCellMatching(tf1, tf2)
    if sum(cellMatching) == cellMatching[1]
        println("fields match (GOOD)")
    else
        println("fields do not match (BAD)")
    end

    if max_error > range * eb
        println("error bound exceeded (BAD)")
    end

    max_error /= range

    println("\tground: $(getTypeFrequencies(tf1))")
    println("\treconstructed: $(getTypeFrequencies(tf2))")
    println("\tfalse cases: $cellMatching")

    println("\nmax error: $max_error")
    println("compression ratio: $ratio")
    println("bitrate: $bitrate")
    println("PSNR: $psnr")
    if compression_time != -1
        println("compression time: $compression_time")
    end
    if decompression_time != -1
        println("decompression time: $decompression_time")
    end

end

function evaluationList2dSymmetric(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, eb, compressed_size::Int64 = -1 )
    tf1 = loadTensorField2dSymmetricFromFolder(ground, dims)
    tf2 = loadTensorField2dSymmetricFromFolder(reconstructed, dims)

    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])

    max_error, range = maxErrorAndRange(tf1, tf2)
    _, mse, frobeniusMse = reconstructionQuality(tf1, tf2)

    cellMatching = topologyCellMatching(tf1,tf2)
    match = sum(cellMatching) == cellMatching[1]

    circularPointDistributionGround = getTypeFrequencies(tf1)
    circularPointDistributionRecon = getTypeFrequencies(tf2)

    if max_error > range * eb
        match = false
    end

    if range != 0.0
        mseByRangeSquared = mse / range^2
        frobeniusMseByRangeSquared = frobeniusMse / range^2
        maxErrorByRange = max_error / range
    else
        mseByRangeSquared = -1.0
        frobeniusMseByRangeSquared = -1.0
        maxErrorByRange = -1.0
    end

    return (match, bitrate, maxErrorByRange, mseByRangeSquared, frobeniusMseByRangeSquared, cellMatching, circularPointDistributionGround, circularPointDistributionRecon)
end