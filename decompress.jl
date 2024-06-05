module decompress

using ..utils
using ..tensorField
using ..huffman

export decompress2d
export decompress2dNaive
export decompress2dSymmetric
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
    θ_bound = reinterpret(Float64, read(vals_file, 8))[1]
    y_bound = reinterpret(Float64, read(vals_file, 8))[1]
    type_indicator = reinterpret(Int64, read(vals_file, 8))[1]
    codes_huffman_length = reinterpret(Int64, read(vals_file, 8))[1]
    codes = huffmanDecode( read(vals_file, codes_huffman_length) )
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
    run(`../SZ3-master/build/bin/sz3 -f -z $output/d.cmp -o $output/d_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $y_bound`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/r.cmp -o $output/r_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $y_bound`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/s.cmp -o $output/s_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $y_bound`)    
    run(`../SZ3-master/build/bin/sz3 -f -z $output/theta.cmp -o $output/theta_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $θ_bound`)

    # Reconstruct
    tf_entries = reconstructEntries2d(dtype, "$output/d_intermediate.dat", "$output/r_intermediate.dat", "$output/s_intermediate.dat", "$output/theta_intermediate.dat", dims, y_bound, codes, losslessStorage, losslessStorage64)

    # Save to file
    for row in 1:2
        for col in 1:2
            saveArray("$output/$decompress_folder/row_$(row)_col_$(col).dat", tf_entries[row, col])
        end
    end

    # Cleanup
    remove("$output/d.cmp")
    remove("$output/r.cmp")
    remove("$output/s.cmp")
    remove("$output/theta.cmp")

    remove("$output/d_intermediate.dat")
    remove("$output/r_intermediate.dat")
    remove("$output/s_intermediate.dat")
    remove("$output/theta_intermediate.dat")

end

function decompress2dSymmetric(compressed_file, decompress_folder, output = "../output")

    # Make output folder for decompression

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
    θ_bound = reinterpret(Float64, read(vals_file, 8))[1]
    r_bound = reinterpret(Float64, read(vals_file, 8))[1]
    trace_bound = reinterpret(Float64, read(vals_file, 8))[1]
    type_indicator = reinterpret(Int64, read(vals_file, 8))[1]
    num_lossless = reinterpret(Int64, read(vals_file, 8))[1]

    if num_lossless != 0
        lossless_storage = reinterpret(Float32, read(vals_file, 4*num_lossless))
        quantization_codes = huffmanDecode(read(vals_file))
    else
        lossless_storage = nothing
        quantization_codes = nothing
    end

    if type_indicator == 1
        dtype = Float64
    else
        dtype = Float32
    end

    close(vals_file)

    run(`../SZ3-master/build/bin/sz3 -f -z $output/theta.cmp -o $output/theta_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $θ_bound`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/r.cmp -o $output/r_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $r_bound`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/trace.cmp -o $output/trace_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $trace_bound`)

    tf_entries = reconstructSymmetricEntries2d(dtype, "$output/theta_intermediate.dat", "$output/r_intermediate.dat", "$output/trace_intermediate.dat", dims, r_bound, lossless_storage, quantization_codes)

    for row in 1:2
        for col in 1:2
            saveArray("$output/$decompress_folder/row_$(row)_col_$(col).dat", tf_entries[row, col])
        end
    end

    remove("$output/r.cmp")
    remove("$output/theta.cmp")
    remove("$output/trace.cmp")

    remove("$output/r_intermediate.dat")
    remove("$output/theta_intermediate.dat")
    remove("$output/trace_intermediate.dat")

end

# Swaps the magnitude of two numbers but preserves their signs.
function signedSwap(a,b)
    return (sign(a)*abs(b), sign(b)*abs(a))
end

function adjustDecompositionEntriesSigns(d_, r_, eVectorGround, eValueGround, y_bound)
    if (eValueGround == COUNTERCLOCKWISE_ROTATION || eVectorGround in [W_RN, PI_BY_4, W_CN]) && r_ < 0
        r_ += y_bound
    elseif (eValueGround == CLOCKWISE_ROTATION || eVectorGround in [W_RS, MINUS_PI_BY_4, W_CS]) && r_ > 0
        r_ -= y_bound
    elseif eValueGround == POSITIVE_SCALING && d_ < 0
        d_ += y_bound
    elseif eValueGround == NEGATIVE_SCALING && d_ > 0
        d_ -= y_bound
    end
    return d_, r_
end

function adjustDecompositionEntries(d, r, s, θ, y_bound, code, decompressing=false)
    if s < 0
        s += y_bound
    end

    code, eVectorGround = getCodeValue( code, CODE_CHANGE_EIGENVECTOR )
    code, eValueGround = getCodeValue( code, CODE_CHANGE_EIGENVALUE )

    if eVectorGround == 0
        eVectorGround = classifyTensorEigenvector(r, s)
    end

    if eValueGround == 0
        eValueGround = classifyTensorEigenvalue(d, r, s)
    end

    d, r = adjustDecompositionEntriesSigns(d, r, eVectorGround, eValueGround, y_bound)

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

    eVectorClassification = classifyTensorEigenvector(r,s)

    return (d,r,s,θ)
end

function reconstructEntries2d(dtype, d_file, r_file, s_file, θ_file, dims, y_bound, codes, losslessStorage, losslessStorage64)

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
    losslessIndex64 = 1

    for i in 1:numPoints
        code = codes[i]
        if code % CODE_LOSSLESS_FULL == 0
            tf_entries[1,1][i] = losslessStorage[losslessIndex]
            tf_entries[1,2][i] = losslessStorage[losslessIndex+1]
            tf_entries[2,1][i] = losslessStorage[losslessIndex+2]
            tf_entries[2,2][i] = losslessStorage[losslessIndex+3]
            losslessIndex += 4
        elseif code % CODE_LOSSLESS_FULL_64 == 0
            tf_entries[1,1][i] = losslessStorage64[losslessIndex64]
            tf_entries[1,2][i] = losslessStorage64[losslessIndex64+1]
            tf_entries[2,1][i] = losslessStorage64[losslessIndex64+2]
            tf_entries[2,2][i] = losslessStorage64[losslessIndex64+3]
            losslessIndex64 += 4
        else

            d_ = d[i]
            r_ = r[i]
            s_ = s[i]
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

            d_, r_, s_, θ_ = adjustDecompositionEntries(d_, r_, s_, θ_ , y_bound, code, true)

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

function reconstructSymmetricEntries2d(dtype, theta_file, r_file, trace_file, dims, r_bound, lossless_storage = nothing, codes = nothing)

    tf_entries = Array{Array{dtype}}(undef, (2,2))
    numPoints = dims[1]*dims[2]*dims[3]
    for row in 1:2
        for col in 1:2
            tf_entries[row,col] = Array{dtype}(undef, (dims...,))
        end
    end

    θ = loadArray(theta_file, Float32)
    r = loadArray(r_file, Float32)
    trace = loadArray(trace_file, Float32)

    lossless_index = 1

    for i in 1:numPoints

        if r[i] <= 0
            r[i] += r_bound
        end

        if !isnothing(codes) && codes[i] == CODE_LOSSLESS_ANGLE
            cplx = r[i] * exp(lossless_storage[lossless_index]*im)
            lossless_index += 1
        else
            cplx = r[i] * exp(θ[i]*im)
        end
        
        Δ = real(cplx)
        off_diagonal = imag(cplx)
        
        tf_entries[1,1][i] = Δ + 0.5*trace[i]
        tf_entries[1,2][i] = off_diagonal
        tf_entries[2,1][i] = off_diagonal
        tf_entries[2,2][i] = -Δ + 0.5*trace[i]

    end

    return tf_entries
end

end