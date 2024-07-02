using ..tensorField

function getTypeFrequencies(tf::SymmetricTensorField2d)
    T,x,y = tf.dims
    types = [0,0,0,0]
    for t in 1:T
        for i in 1:(x-1)
            for j in 1:(y-1)
                for k in 0:1
                    type = getCircularPointType(tf, t, i, j, Bool(k))
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

    T,x,y = tf1.dims
    for j in 1:y
        for i in 1:x
            for t in 1:T

                t1 = getTensor(tf1, t, i, j)
                t2 = getTensor(tf2, t, i, j)

                d1, r1, s1, _ = decomposeTensor(t1)
                d2, r2, s2, _ = decomposeTensor(t2)

                if classifyTensorEigenvector(r1, s1) == classifyTensorEigenvector(r2, s2)
                    result[1,1] += 1
                else
                    result[1,2] += 1
                end

                if classifyTensorEigenvalue(d1, r1, s1) == classifyTensorEigenvalue(d2, r2, s2)
                    result[2,1] += 1
                else
                    result[2,2] += 1
                end
            end
        end
    end

    return result
end

# edge iteration is hardcoded for the specific mesh
function topologyEdgeMatching(tf1::TensorField2d, tf2::TensorField2d)

    # hit edges & miss edges
    result = [0, 0]
    freqs = Dict()

    T, x, y = tf1.dims

    for t in 1:T
        for i1 in 1:x
            for j1 in 1:y
                for edgeClass in 1:3

                    if edgeClass == 1 && i1 != 1
                        i2 = i1 - 1
                        j2 = j1
                    elseif edgeClass == 2 && j1 != 1
                        i2 = i1
                        j2 = j1 - 1
                    elseif edgeClass == 3 && i1 != x && j1 != 1
                        i2 = i1 + 1
                        j2 = j1 - 1
                    else
                        continue
                    end

                    t11 = getTensor(tf1, t, i1, j1)
                    t12 = getTensor(tf2, t, i1, j1)
                    t21 = getTensor(tf1, t, i2, j2)
                    t22 = getTensor(tf2, t, i2, j2)

                    class1 = classifyEdgeEigenvalue(t11, t21)
                    class2 = classifyEdgeEigenvalue(t12, t22)

                    if classifyEdgeEigenvalue(t11, t21) == classifyEdgeEigenvalue(t12, t22)
                        result[1] += 1
                    else
                        result[2] += 1
                        println("$((t, i1, j1)) $((t, i2, j2))")                        
                    end

                end
            end
        end
    end

    things = []
    for k in keys(freqs)
        push!(things, (k, freqs[k]))
    end

    sort!(things, by=f(x)=x[2])
    for t in things
        println(t)
    end

    return result

end

function topologyCellMatching(tf1::TensorField2d, tf2::TensorField2d)

    result = [0,0]

    T, x, y = tf1.dims
    x -= 1
    y -= 1
    for t in 1:T
        for i in 1:x
            for j in 1:y
                for k in 0:1

                    if getCircularPointType(tf1, t, i, j, Bool(k)) == getCircularPointType(tf2, t, i, j, Bool(k))
                        result[1] += 1
                    else
                        result[2] += 1
                    end

                end
            end
        end
    end

    return result

end

function tensorFieldMatchSymmetric(tf1::SymmetricTensorField2d, tf2::SymmetricTensorField2d)
    if tf1.dims != tf2.dims
        return false
    end

    T,x,y = tf1.dims
    for t in 1:T
        for i in 1:(x-1)
            for j in 1:(y-1)
                for k in 0:1
                    type1 = getCircularPointType(tf1, t, i, j, Bool(k))
                    type2 = getCircularPointType(tf2, t, i, j, Bool(k))
                    if type1 != type2
                        println("bad at $t,$i,$j,$k")
                        return false
                    end
                end
            end
        end
    end
    
    return true

end

function asymmetricPSNR(tf_ground::TensorField2d, tf_reconstructed::TensorField2d)
    dtype = typeof(tf_ground.entries[1,1][1,1])
    ground_sos = tf_ground.entries[1,1].^2 + tf_ground.entries[1,2].^2 + tf_ground.entries[2,1].^2 + tf_ground.entries[2,2].^2
    peak_signal = sqrt(maximum(ground_sos)) - sqrt(minimum(ground_sos))

    dims = tf_ground.dims

    mse = sum( (tf_ground.entries[1,1] - tf_reconstructed.entries[1,1]).^2
             + (tf_ground.entries[1,2] - tf_reconstructed.entries[1,2]).^2
             + (tf_ground.entries[2,1] - tf_reconstructed.entries[2,1]).^2
             + (tf_ground.entries[2,2] - tf_reconstructed.entries[2,2]).^2 
    ) / (dims[1]*dims[2]*dims[3])

    psnr = 10 * log(10, peak_signal^2 / mse)
    return psnr

end

function symmetricPSNR(tf_ground::SymmetricTensorField2d, tf_reconstructed::SymmetricTensorField2d)
    dtype = typeof(tf_ground.entries[1,1][1,1])
    ground_sos = tf_ground.entries[1,1].^2 + tf_ground.entries[1,2].^2 + tf_ground.entries[2,2].^2
    peak_signal = sqrt(maximum(ground_sos)) - sqrt(minimum(ground_sos))

    dims = tf_ground.dims

    mse = sum( (tf_ground.entries[1,1] - tf_reconstructed.entries[1,1]).^2
             + (tf_ground.entries[1,2] - tf_reconstructed.entries[1,2]).^2
             + (tf_ground.entries[2,2] - tf_reconstructed.entries[2,2]).^2 
    ) / (dims[1]*dims[2]*dims[3])

    psnr = 10 * log(10, peak_signal^2 / mse)
    return psnr

end

function maxError(tf_ground, tf_reconstructed)
    max_val = -Inf
    min_val = Inf

    max_error = -Inf

    for row in 1:2
        for col in 1:2
            max_val = max(max_val, maximum(tf_ground.entries[row,col]))
            min_val = min(min_val, minimum(tf_ground.entries[row,col]))
            max_error = max(max_error, maximum( abs.(tf_ground.entries[row,col] - tf_reconstructed.entries[row,col] ) ) )
        end
    end

    return (max_error) / (max_val - min_val)

end

function printEvaluation2d(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, entropy::Float64, losslessBitrate::Float64, compressed_size::Int64 = -1, compression_time::Float64 = -1.0, decompression_time::Float64 = -1.0 )
    tf1, dtype = loadTensorField2dFromFolder(ground, dims)
    tf2, _ = loadTensorField2dFromFolder(reconstructed, dims)

    vertexMatching = topologyVertexMatching(tf1, tf2)
    edgeMatching = topologyEdgeMatching(tf1, tf2)
    cellMatching = topologyCellMatching(tf1, tf2)

    if vertexMatching[1,2] == 0 && vertexMatching[2,2] == 0 && edgeMatching[2] == 0 && cellMatching[2] == 0
        result = "GOOD"
    else
        result = "BAD"
    end

    psnr = asymmetricPSNR(tf1, tf2)

    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])

    eltSize = 64
    if dtype == Float32
        eltSize = 32
    end

    ratio = eltSize*4/bitrate

    max_error = maxError(tf1, tf2)

    println("-----------------")
    println(result)
    println("\tvertex eigenvector: ($(vertexMatching[1,1]),$(vertexMatching[1,2]))")
    println("\tvertex eigenvalue: ($(vertexMatching[2,1]), $(vertexMatching[2,2]))")
    println("\tedge eigenvalue ($(edgeMatching[1]), $(edgeMatching[2]))")
    println("\tcell matching ($(cellMatching[1]), $(cellMatching[2]))")

    println("\nmax error: $max_error")
    println("compression ratio: $ratio")
    println("bitrate: $bitrate")
    println("\tentropy: $entropy")
    println("\tlossless: $losslessBitrate")
    println("PSNR: $psnr")
    if compression_time != -1
        println("compression time: $compression_time")
    end
    if decompression_time != -1
        println("decompression time: $decompression_time")
    end

end

function printEvaluation2dSymmetric(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, symmetric::Bool = false, compressed_size::Int64 = -1, compression_time::Float64 = -1.0, decompression_time::Float64 = -1.0 )
    tf1, dtype = loadTensorField2dFromFolder(ground, dims)
    tf2, _ = loadTensorField2dFromFolder(reconstructed, dims)

    if symmetric
        psnr = symmetricPSNR(tf1, tf2)        
    else
        psnr = asymmetricPSNR(tf1, tf2)
    end

    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])

    eltSize = 64
    if dtype == Float32
        eltSize = 32
    end

    numElts = 4
    if symmetric
        numElts = 3
    end

    ratio = eltSize*numElts/bitrate

    max_error = maxError(tf1, tf2)

    println("-----------------")
    if tensorFieldMatchSymmetric(tf1, tf2)
        println("fields match (GOOD)")
    else
        println("fields do not match (BAD)")
    end

    println("\tground: $(getTypeFrequencies(tf1))")
    println("\treconstructed: $(getTypeFrequencies(tf2))")

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