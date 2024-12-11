using ..tensorField

function getTypeFrequencies(tf::TensorField2d)
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

                if classifyTensorEigenvector(r1, s1) == classifyTensorEigenvector(r2, s2)
                    result[1,1] += 1
                else
                    # println((i,j,t))
                    result[1,2] += 1
                end

                if classifyTensorEigenvalue(d1, r1, s1) == classifyTensorEigenvalue(d2, r2, s2)
                    result[2,1] += 1
                else
                    # println((i,j,t))
                    result[2,2] += 1
                end

            end
        end
    end

    return result
end

# edge iteration is hardcoded for the specific mesh
function topologyEdgeMatching(tf1::TensorField2d, tf2::TensorField2d, edgeEB, eigenvalue = true, eigenvector = true)

    # hit edges & miss edges
    result = [0, 0]
    freqs = Dict()

    x, y, T = tf1.dims

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

                    t11 = getTensor(tf1, i1, j1, t)
                    t12 = getTensor(tf2, i1, j1, t)
                    t21 = getTensor(tf1, i2, j2, t)
                    t22 = getTensor(tf2, i2, j2, t)

                    if edgesMatch( t11, t21, t12, t22, edgeEB, eigenvalue, eigenvector )
                        result[1] += 1
                    else
                        result[2] += 1
                        # println((i1,j1,t,i2,j2,t))
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

    return result

end

function topologyCellMatching(tf1::TensorField2d, tf2::TensorField2d)

    result = [0,0]

    x, y, T = tf1.dims
    x -= 1
    y -= 1
    for t in 1:T
        for i in 1:x
            for j in 1:y
                for k in 0:1

                    if getCircularPointType(tf1, i, j, t, Bool(k)) == getCircularPointType(tf2, i, j, t, Bool(k))
                        result[1] += 1
                    else
                        result[2] += 1
                        # println((i,j,t,Bool(k)))
                    end

                end
            end
        end
    end

    return result

end

function tensorFieldMatchSymmetric(tf1::TensorField2d, tf2::TensorField2d)
    if tf1.dims != tf2.dims
        return false
    end

    numFC = 0

    x,y,T = tf1.dims
    for t in 1:T
        for i in 1:(x-1)
            for j in 1:(y-1)
                for k in 0:1
                    type1 = getCircularPointType(tf1, i, j, t, Bool(k))
                    type2 = getCircularPointType(tf2, i, j, t, Bool(k))
                    if type1 != type2
                        numFC += 1
                    end
                end
            end
        end
    end
    
    if numFC != 0
        println("$numFC false cases")
        return false
    end

    return true

end

function asymmetricPSNR(tf_ground::TensorField2d, tf_reconstructed::TensorField2d)
    return -1.0
end

function symmetricPSNR(tf_ground::TensorField2d, tf_reconstructed::TensorField2d)
    return -1.0
end

function maxErrorAndRange(tf_ground, tf_reconstructed)

    max_val = maximum(tf_ground.entries)
    min_val = minimum(tf_ground.entries)
    max_error = maximum( abs.(tf_ground.entries - tf_reconstructed.entries) )

    return (max_error), (max_val - min_val)

end

function printEvaluation2d(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, entropy::Float64, losslessBitrate::Float64, compressed_size::Int64 = -1, compression_time::Float64 = -1.0, decompression_time::Float64 = -1.0, edgeEB = 1.0, eigenvalue = true, eigenvector = true)
    tf1 = loadTensorField2dFromFolder(ground, dims)
    tf2 = loadTensorField2dFromFolder(reconstructed, dims)

    vertexMatching = topologyVertexMatching(tf1, tf2)
    edgeMatching = topologyEdgeMatching(tf1, tf2, edgeEB, eigenvalue, eigenvector)
    cellMatching = topologyCellMatching(tf1, tf2)

    if vertexMatching[1,2] == 0 && vertexMatching[2,2] == 0 && edgeMatching[2] == 0 && cellMatching[2] == 0
        result = "GOOD"
    else
        result = "BAD"
    end

    psnr = asymmetricPSNR(tf1, tf2)

    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])

    eltSize = 64

    ratio = eltSize*4/bitrate

    max_error, range = maxErrorAndRange(tf1, tf2)
    max_error /= range

    println("-----------------")
    println(result)
    println("\tvertex eigenvector: ($(vertexMatching[1,1]),$(vertexMatching[1,2]))")
    println("\tvertex eigenvalue: ($(vertexMatching[2,1]), $(vertexMatching[2,2]))")
    println("\tedges: ($(edgeMatching[1]), $(edgeMatching[2]))")
    println("\tcell matching: ($(cellMatching[1]), $(cellMatching[2]))")

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

function evaluationList2d(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, compressed_size::Int64 = -1, edgeEB = 1.0, eigenvalue = true, eigenvector = true)
    tf1 = loadTensorField2dFromFolder(ground, dims)
    tf2 = loadTensorField2dFromFolder(reconstructed, dims)

    vertexMatching = topologyVertexMatching(tf1, tf2)
    edgeMatching = topologyEdgeMatching(tf1, tf2, edgeEB, eigenvalue, eigenvector)
    cellMatching = topologyCellMatching(tf1, tf2)

    if (!eigenvector || vertexMatching[1,2] == 0) && (!eigenvalue || vertexMatching[2,2] == 0) && edgeMatching[2] == 0 && (!eigenvector || cellMatching[2] == 0)
        preserved = true
    else
        preserved = false
    end

    # psnr = asymmetricPSNR(tf1, tf2) # going to ignore this for now...
    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])
    max_error, range = maxErrorAndRange(tf1, tf2)

    return (preserved, max_error, range, bitrate, vertexMatching[1,2], vertexMatching[2,2], edgeMatching[2], cellMatching[2])

end

function printEvaluation2dSymmetric(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, symmetric::Bool = false, compressed_size::Int64 = -1, compression_time::Float64 = -1.0, decompression_time::Float64 = -1.0 )
    tf1 = loadTensorField2dFromFolder(ground, dims)
    tf2 = loadTensorField2dFromFolder(reconstructed, dims)

    if symmetric
        psnr = symmetricPSNR(tf1, tf2)
    else
        psnr = asymmetricPSNR(tf1, tf2)
    end

    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])

    eltSize = 64

    numElts = 4
    if symmetric
        numElts = 3
    end

    ratio = eltSize*numElts/bitrate

    max_error, range = maxErrorAndRange(tf1, tf2)
    max_error /= range

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

function evaluationList2dSymmetric(ground::String, reconstructed::String, dims::Tuple{Int64, Int64, Int64}, compressed_size::Int64 = -1 )
    tf1 = loadTensorField2dFromFolder(ground, dims)
    tf2 = loadTensorField2dFromFolder(reconstructed, dims)

    bitrate = compressed_size*8/(dims[1]*dims[2]*dims[3])

    max_error, range = maxErrorAndRange(tf1, tf2)

    match = tensorFieldMatchSymmetric(tf1, tf2)

    return (match, max_error, range, bitrate)

end