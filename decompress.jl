module decompress

using ..utils
using ..tensorField
using ..huffman

export decompress2d
export decompress2dNaive
export decompress2dSymmetric
export decompress2dSymmetricNaive

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

function decompress2d(compressed_file, decompress_folder, output = "../output")

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

    # Read in quantization bytes
    baseCodeBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    baseCodeBytes = read(vals_file, baseCodeBytesLength)
    θAndSFixBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    θAndSFixBytes = read(vals_file, θAndSFixBytesLength)
    dBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    dBytes = read(vals_file, dBytesLength)
    rBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    rBytes = read(vals_file, rBytesLength)
    sBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    sBytes = read(vals_file, sBytesLength)
    eigenvectorSpecialCaseLength = reinterpret(Int64, read(vals_file, 8))[1]
    eigenvectorSpecialCaseBytes = read(vals_file, eigenvectorSpecialCaseLength)

    # Read in various lossless lists
    losslessdLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_d = reinterpret(Float64, read(vals_file, losslessdLength))
    losslessrLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_r = reinterpret(Float64, read(vals_file, losslessrLength))
    losslesssLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_s = reinterpret(Float64, read(vals_file, losslesssLength))
    losslessθLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_θ = reinterpret(Float64, read(vals_file, losslessθLength))
    losslessALength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_A = reinterpret(Float64, read(vals_file, losslessALength))
    losslessBLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_B = reinterpret(Float64, read(vals_file, losslessBLength))
    losslessCLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_C = reinterpret(Float64, read(vals_file, losslessCLength))
    losslessDLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_D = reinterpret(Float64, read(vals_file, losslessDLength))

    close(vals_file)

    # De-Huffman the codes
    baseCodes = reshape(huffmanDecode(baseCodeBytes),Tuple(dims))
    θAndSFixCodes = reshape(huffmanDecode(θAndSFixBytes),Tuple(dims))
    dCodes = reshape(huffmanDecode(dBytes),Tuple(dims))
    rCodes = reshape(huffmanDecode(rBytes),Tuple(dims))
    sCodes = reshape(huffmanDecode(sBytes),Tuple(dims))

    eigenvectorSpecialCaseArray = huffmanDecode(eigenvectorSpecialCaseBytes)
    if length(eigenvectorSpecialCaseArray) == 0
        eigenvectorSpecialCaseCodes = zeros(Int64, Tuple(dims))
    else
        eigenvectorSpecialCaseCodes = reshape(eigenvectorSpecialCaseArray,Tuple(dims))
    end

    # Decompress from SZ and load into a tensor field
    run(`../SZ3-master/build/bin/sz3 -f -z $output/row_1_col_1.cmp -o $output/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/row_1_col_2.cmp -o $output/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/row_2_col_1.cmp -o $output/row_2_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)    
    run(`../SZ3-master/build/bin/sz3 -f -z $output/row_2_col_2.cmp -o $output/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)

    dims_tuple::Tuple{Int64, Int64, Int64} = (dims[1], dims[2], dims[3])
    tf, dtype = loadTensorField2dFromFolder("$output", dims_tuple)

    next_lossless_full = 1
    next_lossless_d = 1
    next_lossless_r = 1
    next_lossless_s = 1
    next_lossless_θ = 1

    tfTest, dtype2 = loadTensorField2dFromFolder("$output/test", dims_tuple)

    # Iterate through the cells and adjust accordingly.
    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]

                # Make sure that a change must actually be applied in the first place.
                if !(baseCodes[t,i,j] == 0 && θAndSFixCodes[t,i,j] == 0 && dCodes[t,i,j] == 127 && rCodes[t,i,j] == 127 && sCodes[t,i,j] == 127)
                    precision::UInt8 = baseCodes[t,i,j] >> 4

                    if precision >= 8
                        
                        setTensor( tf, t, i, j, [ lossless_A[next_lossless_full] lossless_B[next_lossless_full] ; lossless_C[next_lossless_full] lossless_D[next_lossless_full] ] )
                        next_lossless_full += 1

                    else
                        swapCode::UInt8 = baseCodes[t,i,j] & (2^4-1)
                        θCode::UInt8 = θAndSFixCodes[t,i,j] & (2^6-1)
                        sFix::UInt8 = ( θAndSFixCodes[t,i,j] & (2^6+2^7) ) >> 6
                        eigenvectorSpecialCaseCode::UInt8 = eigenvectorSpecialCaseCodes[t,i,j]

                        tensor = getTensor(tf, t, i, j)
                        d,r,s,θ = decomposeTensor(tensor)

                        if dCodes[t,i,j] == 255
                            d = lossless_d[next_lossless_d]
                            next_lossless_d += 1
                        else
                            d = d + aeb * (dCodes[t,i,j] - 127) / (2^precision)
                        end

                        if rCodes[t,i,j] == 255
                            r = lossless_r[next_lossless_r]
                            next_lossless_r += 1
                        else
                            r = r + aeb * (rCodes[t,i,j] - 127) / (2^precision)
                        end

                        if sCodes[t,i,j] == 255
                            s = lossless_s[next_lossless_s]
                            next_lossless_s += 1
                        else
                            s = s + sqrt(2) * aeb * (sCodes[t,i,j] - 127) / (2^precision)
                        end

                        if θCode == 2^6-1
                            θ = lossless_θ[next_lossless_θ]
                            next_lossless_θ += 1
                        else
                            θ = θ + 2pi / (2^6-1) * θCode
                        end

                        # apply the various s fixes
                        if s < -(sqrt(2)-1)*aeb
                            s += sqrt(2)*aeb
                        elseif s < 0
                            s += (sqrt(2)-1)*aeb
                        end

                        if sFix == 1
                            s -= (sqrt(2)-1)*aeb
                        elseif sFix == 2
                            s += (sqrt(2)-1)*aeb
                        end

                        # apply the swapping
                        d_sign_swap = Bool((swapCode & (1 << 3)) >> 3)
                        r_sign_swap = Bool((swapCode & (1 << 2)) >> 2)
                        d_largest_swap = Bool((swapCode & (1 << 1)) >> 1)
                        r_over_s_swap = Bool(swapCode & 1)

                        if d_sign_swap
                            if d < 0
                                d += aeb
                            else
                                d -= aeb
                            end
                        end

                        if r_sign_swap
                            if r < 0
                                r += aeb
                            else
                                r -= aeb
                            end
                        end

                        if d_largest_swap
                            # swap d with the larger of r or s
                            if abs(r) > s
                                sign_d = (d >= 0) ? 1 : -1
                                sign_r = (r >= 0) ? 1 : -1
                                temp = d
                                d = sign_d * abs(r)
                                r = sign_r * abs(temp)
                            else
                                sign_d = (d >= 0) ? 1 : -1
                                temp = d
                                d = sign_d * s
                                s = abs(temp)
                            end
                        end

                        if eigenvectorSpecialCaseCode == 0
                            if r_over_s_swap
                                sign_r = (r >= 0) ? 1 : -1
                                temp = r
                                r = sign_r * s
                                s = abs(temp)
                            end
                        elseif eigenvectorSpecialCaseCode == 1
                            sign_r = (r >= 0) ? 1 : -1
                            r = sign_r * s
                        else
                            r = 0
                        end

                        setTensor(tf, t, i, j, recomposeTensor(d, r, s, θ))

                    end # end if precision >= 8 (then else for the main reconstruction)

                end # end if not all of the base codes are 0

            end # end for
        end # end for 
    end # end for

    numMatchFinal = 0
    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]
                if getTensor(tfTest, t, i, j) != getTensor(tf, t, i, j)
                    println((t,i,j))
                    println(getTensor(tfTest, t, i, j))
                    println(getTensor(tf, t, i, j))
                    # exit()
                else
                    numMatchFinal += 1
                end
            end
        end
    end

    # Save to file
    for row in 1:2
        for col in 1:2
            saveArray("$output/$decompress_folder/row_$(row)_col_$(col).dat", tf.entries[row, col])
        end
    end

    remove("$output/row_1_col_1.dat")
    remove("$output/row_1_col_2.dat")
    remove("$output/row_2_col_1.dat")
    remove("$output/row_2_col_2.dat")
    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_1.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/vals.bytes")
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
                intermediateTensor = getTensor(tf, 1, i, j)
                t = (intermediateTensor[1,1]+intermediateTensor[2,2])/2
                nextTensor = [ losslessValues[next_lossless]+t losslessValues[next_lossless+1] ; losslessValues[next_lossless+1] -losslessValues[next_lossless]+t ]
                setTensor(tf, 1, i, j, nextTensor)
                next_lossless += 2
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