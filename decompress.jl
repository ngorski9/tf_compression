module decompress

using ..utils
using ..tensorField
using ..huffman

export decompress2d
export decompress2dNaive
export decompress2dSymmetric
export decompress2dSymmetricOld
export decompress2dSymmetricNaive
export decompress2dSymmetricSimple
export reconstructSymmetricEntries2d
export adjustDecompositionEntries
export adjustDecompositionEntriesSigns

function decompress2dNaive(compressed_file, decompress_folder, output = "../output")
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`tar xvf $output/$compressed_file.tar`)

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    close(vals_file)

    run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    run(`../SZ3-master/build/bin/sz3 -d -z $output/row_2_col_1.cmp -o $output/$decompress_folder/row_2_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    run(`../SZ3-master/build/bin/sz3 -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_1.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")

end

function decompress2dSymmetricNaive(compressed_file, decompress_folder, output = "../output")
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`tar xvf $output/$compressed_file.tar`)

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    close(vals_file)

    run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    run(`../SZ3-master/build/bin/sz3 -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    run(`cp $output/$decompress_folder/row_1_col_2.dat $output/$decompress_folder/row_2_col_1.dat`)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")

end

function decompress2d(compressed_file, decompress_folder, output="../output")

    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`tar xvf $output/$compressed_file.tar`)

    cd(cwd)

    # Read in metadata
    vals_file = open("$output/vals.bytes")
    dims = reinterpret(Int64, read(vals_file, 24))
    aeb = reinterpret(Float64, read(vals_file, 8))[1]
    type_indicator = reinterpret(Int64, read(vals_file, 8))[1]
    codes_huffman_length = reinterpret(Int64, read(vals_file, 8))[1]
    codes = huffmanDecode( read(vals_file, codes_huffman_length) )
    quantization_huffman_length = reinterpret(Int64, read(vals_file, 8))[1]
    quantization = huffmanDecode( read(vals_file, quantization_huffman_length) )
    lossless_storage_length = reinterpret(Int64, read(vals_file, 8))[1]
    losslessStorage = reinterpret(Float32, read(vals_file, 4*lossless_storage_length))
    lossless_storage_length_64 = reinterpret(Int64, read(vals_file,8))[1]
    losslessStorage64 = reinterpret(Float64, read(vals_file, 8*lossless_storage_length_64))

    if type_indicator == 1
        dtype = Float64
    else
        dtype = Float32
    end

    close(vals_file)

    # Decompress
    run(`../SZ3-master/build/bin/sz3 -f -z $output/a.cmp -o $output/a_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/b.cmp -o $output/b_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/c.cmp -o $output/c_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)    
    run(`../SZ3-master/build/bin/sz3 -f -z $output/d.cmp -o $output/d_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)

    # Reconstruct
    tf_entries = reconstructEntries2d(dtype, "$output/a_intermediate.dat", "$output/b_intermediate.dat", "$output/c_intermediate.dat", "$output/d_intermediate.dat", dims, aeb, codes, quantization, losslessStorage, losslessStorage64)

    # Save to file
    for row in 1:2
        for col in 1:2
            saveArray("$output/$decompress_folder/row_$(row)_col_$(col).dat", tf_entries[row, col])
        end
    end

    # Cleanup
    remove("$output/a.cmp")
    remove("$output/b.cmp")
    remove("$output/c.cmp")
    remove("$output/d.cmp")

    remove("$output/a_intermediate.dat")
    remove("$output/b_intermediate.dat")
    remove("$output/c_intermediate.dat")
    remove("$output/d_intermediate.dat")

end

# Swaps the magnitude of two numbers but preserves their signs.
function signedSwap(a,b)
    return (sign(a)*abs(b), sign(b)*abs(a))
end

function adjustDecompositionEntriesSigns(d_, r_, eVectorGround, eValueGround, aeb)
    if (eValueGround == COUNTERCLOCKWISE_ROTATION || eVectorGround in [W_RN, PI_BY_4, W_CN]) && r_ < 0
        r_ += aeb
    elseif (eValueGround == CLOCKWISE_ROTATION || eVectorGround in [W_RS, MINUS_PI_BY_4, W_CS]) && r_ > 0
        r_ -= aeb
    elseif eValueGround == POSITIVE_SCALING && d_ < 0
        d_ += aeb
    elseif eValueGround == NEGATIVE_SCALING && d_ > 0
        d_ -= aeb
    end
    return d_, r_
end

function adjustDecompositionEntries(d, r, s, θ, aeb, code, decompressing=false)

    code, eVectorGround = getCodeValue( code, CODE_CHANGE_EIGENVECTOR )
    code, eValueGround = getCodeValue( code, CODE_CHANGE_EIGENVALUE )

    if eVectorGround == 0
        eVectorGround = classifyTensorEigenvector(r, s)
    end

    if eValueGround == 0
        eValueGround = classifyTensorEigenvalue(d, r, s)
    end

    d, r = adjustDecompositionEntriesSigns(d, r, eVectorGround, eValueGround, aeb)

    if eValueGround == ANISOTROPIC_STRETCHING
        if s < abs(r)
            r,s = signedSwap(r,s)
        end

        if s < abs(d)
            d,s = signedSwap(d,s)
        end
    elseif eValueGround == CLOCKWISE_ROTATION || eValueGround == COUNTERCLOCKWISE_ROTATION
        if abs(r) < abs(s)
            r,s = signedSwap(r,s)
        end

        if abs(r) < abs(d)
            d,r = signedSwap(d,r)
        end
    else
        # In this case, we have d as the largest of the 3,
        # and the order of the other 2 depends on eigenvector classification
       
        if abs(d) < s
            d,s = signedSwap(d,s)
        end

        if abs(d) < abs(r)
            d,r = signedSwap(d,r)
        end

        if ((eVectorGround == W_RN || eVectorGround == W_RS) && s < abs(r)) || ((eVectorGround == W_CN || eVectorGround == W_CS) && abs(r) < s)
            r,s = signedSwap(r,s)
        end
    end

    return (d,r,s,θ)
end

function reconstructEntries2d(dtype, a_file, b_file, c_file, d_file, dims, aeb, codes, quantization, losslessStorage, losslessStorage64)

    tf_entries = Array{Array{dtype}}(undef, (2,2))    
    numPoints = dims[1]*dims[2]*dims[3]

    for row in 1:2
        for col in 1:2
            tf_entries[row,col] = Array{dtype}(undef, (dims...))
        end
    end

    A = loadArray(a_file, Float32)
    B = loadArray(b_file, Float32)
    C = loadArray(c_file, Float32)
    D = loadArray(d_file, Float32)

    flatDims = size(A)

    d = zeros(Float32, flatDims)
    r = zeros(Float32, flatDims)
    s = zeros(Float32, flatDims)
    θ = zeros(Float32, flatDims)

    for i in 1:numPoints
        tensor = [A[i] B[i] ; C[i] D[i]]
        d_, r_, s_, θ_ = decomposeTensor(tensor)

        d[i] = d_
        r[i] = r_
        if s_ < 0
            s[i] = s_ + aeb
        else
            s[i] = s_
        end
        θ[i] = θ_
    end

    losslessIndex = 1
    losslessIndex64 = 1
    numLossless64 = 0

    for i in 1:numPoints
        code = codes[i]
        if code % CODE_LOSSLESS_FULL == 0
            tf_entries[1,1][i] = losslessStorage[losslessIndex]
            tf_entries[1,2][i] = losslessStorage[losslessIndex+1]
            tf_entries[2,1][i] = losslessStorage[losslessIndex+2]
            tf_entries[2,2][i] = losslessStorage[losslessIndex+3]
            losslessIndex += 4
        elseif code % CODE_LOSSLESS_FULL_64 == 0
            numLossless64 += 1
            tf_entries[1,1][i] = losslessStorage64[losslessIndex64]
            tf_entries[1,2][i] = losslessStorage64[losslessIndex64+1]
            tf_entries[2,1][i] = losslessStorage64[losslessIndex64+2]
            tf_entries[2,2][i] = losslessStorage64[losslessIndex64+3]
            losslessIndex64 += 4
        else

            d_ = d[i] + 2 * aeb * quantization[i] / (2^MAX_PRECISION)
            r_ = r[i] + 2 * aeb * quantization[numPoints + i] / (2^MAX_PRECISION)
            s_ = s[i] + 2 * aeb * quantization[2*numPoints + i] / (2^MAX_PRECISION)
            θ_ = θ[i]

            if code % CODE_LOSSLESS_D == 0
                d_ = losslessStorage[losslessIndex]
                losslessIndex += 1
            end

            if code % CODE_LOSSLESS_R == 0
                r_ = losslessStorage[losslessIndex]
                losslessIndex += 1
            end
        
            if code % CODE_LOSSLESS_S == 0
                s_ = losslessStorage[losslessIndex]
                losslessIndex += 1
            end

            if code % CODE_LOSSLESS_ANGLE == 0
                θ_ = losslessStorage[losslessIndex]
                losslessIndex += 1
            end

            d_, r_, s_, θ_ = adjustDecompositionEntries(d_, r_, s_, θ_ , aeb, code, true)

            recomposition = recomposeTensor(d_, r_, s_, θ_)

            tf_entries[1,1][i] = recomposition[1,1]
            tf_entries[1,2][i] = recomposition[1,2]
            tf_entries[2,1][i] = recomposition[2,1]
            tf_entries[2,2][i] = recomposition[2,2]
        end

    end

    return tf_entries

end

function reconstructEntries2dEigenvector(dtype, d_file, r_file, s_file, θ_file, dims, y_bound, codes, losslessStorage)

    tf_entries = Array{Array{dtype}}(undef, (2,2))
    numPoints = dims[1]*dims[2]*dims[3]

    for row in 1:2
        for col in 1:2
            tf_entries[row,col] = Array{dtype}(undef, (dims...))
        end
    end

    d = loadArray(d_file, Float32)
    r = loadArray(r_file, Float32)
    s = loadArray(s_file, Float32)
    θ = loadArray(θ_file, Float32)

    losslessIndex = 1

    for i in 1:numPoints
        code = codes[i]
        if code % CODE_LOSSLESS_FULL == 0
            tf_entries[1,1][i] = losslessStorage[losslessIndex]
            tf_entries[1,2][i] = losslessStorage[losslessIndex+1]
            tf_entries[2,1][i] = losslessStorage[losslessIndex+2]
            tf_entries[2,2][i] = losslessStorage[losslessIndex+3]
        else

            d_ = d[i]
            r_ = r[i]
            s_ = s[i]
            θ_ = θ[i]

            if code % CODE_LOSSLESS_R == 0
                r_ = losslessStorage[losslessIndex]
                losslessIndex += 1
            end
        
            if code % CODE_LOSSLESS_S == 0
                s_ = losslessStorage[losslessIndex]
                losslessIndex += 1
            end

            d_, r_, s_, θ_ = adjustDecompositionEntries(d_, r_, s_, θ_ , y_bound, code)

            recomposition = recomposeTensor(d_, r_, s_, θ_)

            tf_entries[1,1][i] = recomposition[1,1]
            tf_entries[1,2][i] = recomposition[1,2]
            tf_entries[2,1][i] = recomposition[2,1]
            tf_entries[2,2][i] = recomposition[2,2]
        end

    end

    return tf_entries

end

function decompress2dSymmetric(compressed_file, decompress_folder, bits, output = "../output")
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`tar xvf $output/$compressed_file.tar`)

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    codeBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    codeBytes = read(vals_file, codeBytesLength)
    losslessValues = reinterpret(Float64, read(vals_file))
    close(vals_file)

    codes = reshape(huffmanDecode(codeBytes),Tuple(dims))

    run(`../SZ3-master/build/bin/sz3 -f -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    run(`cp $output/$decompress_folder/row_1_col_2.dat $output/$decompress_folder/row_2_col_1.dat`)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")
    dims_tuple::Tuple{Int64, Int64, Int64} = (dims[1], dims[2], dims[3])
    tf, dtype = loadTensorField2dFromFolder("$output/$decompress_folder", dims_tuple)

    # adjust

    next_lossless = 1
    for j in 1:dims[3]
        for i in 1:dims[2]
            if codes[1,i,j] == 2^bits-1
                nextTensor = [ losslessValues[next_lossless] losslessValues[next_lossless + 1] ; losslessValues[next_lossless + 1] losslessValues[next_lossless + 2] ]
                setTensor(tf, 1, i, j, nextTensor)
                next_lossless += 3
            elseif codes[1,i,j] != 0
                t, r, θ = decomposeTensorSymmetric( getTensor( tf, 1, i, j ) )
                θ += 2pi/(2^bits-1)*codes[1,i,j]
                setTensor(tf, 1, i, j, recomposeTensorSymmetric( t, r, θ ))
            end
        end
    end
    # Save to file
    for row in 1:2
        for col in 1:2
            saveArray("$output/$decompress_folder/row_$(row)_col_$(col).dat", tf.entries[row, col])
        end
    end

end

end