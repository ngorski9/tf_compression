module decompress

using StaticArrays

using ..utils
using ..tensorField
using ..huffman

export decompress2d
export decompress2dNaive
export decompress2dSymmetric
export decompress2dSymmetricNaive
export decompress2dSymmetricNaiveWithMask

function decompress2dNaive(compressed_file, decompress_folder, output = "../output", baseCompressor = "sz3")
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $output/$compressed_file.tar.zst`)
    run(`tar xvf $output/$compressed_file.tar`)

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    close(vals_file)

    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[3]) $(dims[2]) $(dims[1]) -M ABS $bound`)
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[3]) $(dims[2]) $(dims[1]) -M ABS $bound`)
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_2_col_1.cmp -o $output/$decompress_folder/row_2_col_1.dat -3 $(dims[3]) $(dims[2]) $(dims[1]) -M ABS $bound`)
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[3]) $(dims[2]) $(dims[1]) -M ABS $bound`)
    elseif baseCompressor == "mgard"
        run(`../MGARD/build/bin/mgard-cpu -x -t d -c $output/row_1_col_1.cmp -d $output/$decompress_folder/row_1_col_1.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $bound -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t d -c $output/row_1_col_2.cmp -d $output/$decompress_folder/row_1_col_2.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $bound -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t d -c $output/row_2_col_1.cmp -d $output/$decompress_folder/row_2_col_1.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $bound -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t d -c $output/row_2_col_2.cmp -d $output/$decompress_folder/row_2_col_2.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $bound -s $smoothness`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_1.dat --pwe $bound`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_2.dat --pwe $bound`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_1.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_2_col_1.dat --pwe $bound`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_2_col_2.dat --pwe $bound`)
    elseif baseCompressor == "zfp"
        run(`../zfp/build/bin/zfp -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $bound`)
        run(`../zfp/build/bin/zfp -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $bound`)
        run(`../zfp/build/bin/zfp -d -z $output/row_2_col_1.cmp -o $output/$decompress_folder/row_2_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $bound`)
        run(`../zfp/build/bin/zfp -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $bound`)
    end

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_1.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")
    remove("$output/vals.bytes")

    return [0.0]
end

function decompress2dSymmetricNaive(compressed_file, decompress_folder, output = "../output", baseCompressor = "sz3")
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $output/$compressed_file.tar.zst`)
    run(`tar xvf $output/$compressed_file.tar`)

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    close(vals_file)

    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_1.dat --pwe $bound`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_2.dat --pwe $bound`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_2_col_1.dat --pwe $bound`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end        

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")
    remove("$output/vals.bytes")

    return [0.0]
end

function decompress2dSymmetricNaiveWithMask(compressed_file, decompress_folder, output = "../output", baseCompressor = "sz3")
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $output/$compressed_file.tar.zst`)
    run(`tar xvf $output/$compressed_file.tar`)

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    huffmanBytes = read(vals_file)
    close(vals_file)

    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3-master/build/bin/sz3 -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_1.dat --pwe $bound`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_2.dat --pwe $bound`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_2_col_1.dat --pwe $bound`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end

    mask = huffmanDecode(huffmanBytes)

    if length(mask) > 0
        tf = loadTensorField2dSymmetricFromFolder("$output/$decompress_folder", (dims[1]*dims[2]*dims[3],1,1))

        for i in 1:(dims[1]*dims[2]*dims[3])
            if mask[i] == 0.0
                tf.entries[1,i,1,1] = 0.0
                tf.entries[2,i,1,1] = 0.0
                tf.entries[3,i,1,1] = 0.0
            end
        end

        saveTensorFieldSymmetric64("$output/$decompress_folder", tf)        
    end

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")
    remove("$output/vals.bytes")

    return [0.0]
end

function decompress2d(compressed_file, decompress_folder, output = "../output", baseCompressor = "sz3", parameter = 1.0)
    startTime = time()

    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $output/$compressed_file.tar.zst`)
    zstdSplit = time()
    run(`tar xvf $output/$compressed_file.tar`)
    tarSplit = time()

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

    dims_tuple = Tuple(dims)

    # De-Huffman the codes
    baseCodesBase = huffmanDecode(baseCodeBytes)
    if length(baseCodesBase) == 0
        baseCodes = zeros(Int64, dims_tuple)
    else
        baseCodes = reshape(baseCodesBase,dims_tuple)
    end

    θAndSFixCodesBase = huffmanDecode(θAndSFixBytes)
    if length(θAndSFixCodesBase) == 0
        θAndSFixCodes = zeros(Int64, dims_tuple)
    else
        θAndSFixCodes = reshape(θAndSFixCodesBase, dims_tuple)
    end

    dCodesBase = huffmanDecode(dBytes)
    if length(dCodesBase) == 0
        dCodes = 127*ones(Int64, dims_tuple)
    else
        dCodes = reshape(dCodesBase, dims_tuple)
    end

    rCodesBase = huffmanDecode(rBytes)
    if length(rCodesBase) == 0
        rCodes = 127*ones(Int64, dims_tuple)
    else
        rCodes = reshape(rCodesBase, dims_tuple)
    end

    sCodesBase = huffmanDecode(sBytes)
    if length(sCodesBase) == 0
        sCodes = 127*ones(Int64, dims_tuple)
    else
        sCodes = reshape(sCodesBase, dims_tuple)
    end

    eigenvectorSpecialCaseArray = huffmanDecode(eigenvectorSpecialCaseBytes)
    if length(eigenvectorSpecialCaseArray) == 0
        eigenvectorSpecialCaseCodes = zeros(Int64, dims_tuple)
    else
        eigenvectorSpecialCaseCodes = reshape(eigenvectorSpecialCaseArray, dims_tuple)
    end

    readSplit = time()

    # Decompress from SZ and load into a tensor field
    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -f -z $output/row_1_col_1.cmp -o $output/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $(parameter*aeb)`)
        run(`../SZ3-master/build/bin/sz3 -f -z $output/row_1_col_2.cmp -o $output/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $(parameter*aeb)`)
        run(`../SZ3-master/build/bin/sz3 -f -z $output/row_2_col_1.cmp -o $output/row_2_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $(parameter*aeb)`)    
        run(`../SZ3-master/build/bin/sz3 -f -z $output/row_2_col_2.cmp -o $output/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $(parameter*aeb)`)
    elseif baseCompressor == "mgard"
        run(`../MGARD/build/bin/mgard-cpu -x -t s -c $output/row_1_col_1.cmp -d $output/row_1_col_1.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t s -c $output/row_1_col_2.cmp -d $output/row_1_col_2.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t s -c $output/row_2_col_1.cmp -d $output/row_2_col_1.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t s -c $output/row_2_col_2.cmp -d $output/row_2_col_2.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_1_col_1.dat --pwe $(parameter*aeb)`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_1_col_2.dat --pwe $(parameter*aeb)`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_1.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_2_col_1.dat --pwe $(parameter*aeb)`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_2_col_2.dat --pwe $(parameter*aeb)`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end

    baseDecompressorSplit = time()

    dims_tuple::Tuple{Int64, Int64, Int64} = (dims[1], dims[2], dims[3])
    tf = loadTensorField2dFromFolder("$output", dims_tuple)

    read2Split = time()

    next_lossless_full = 1
    next_lossless_d = 1
    next_lossless_r = 1
    next_lossless_s = 1
    next_lossless_θ = 1

    # Iterate through the cells and adjust accordingly.
    for t in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]

                # Make sure that a change must actually be applied in the first place.
                if !(baseCodes[i,j,t] == 0 && θAndSFixCodes[i,j,t] == 0 && dCodes[i,j,t] == 127 && rCodes[i,j,t] == 127 && sCodes[i,j,t] == 127)
                    precision::UInt8 = baseCodes[i,j,t] >> 4

                    if precision >= 8
                        setTensor( tf, i, j, t,  SMatrix{2,2,Float64}(lossless_A[next_lossless_full], lossless_C[next_lossless_full], lossless_B[next_lossless_full], lossless_D[next_lossless_full]) )
                        next_lossless_full += 1

                    else
                        swapCode::UInt8 = baseCodes[i,j,t] & (2^4-1)
                        θCode::UInt8 = θAndSFixCodes[i,j,t] & (2^6-1)
                        sFix::UInt8 = ( θAndSFixCodes[i,j,t] & (2^6+2^7) ) >> 6
                        eigenvectorSpecialCaseCode::UInt8 = eigenvectorSpecialCaseCodes[i,j,t]

                        tensor = getTensor(tf, i, j, t)
                        d,r,s,θ = decomposeTensor(tensor)

                        if dCodes[i,j,t] == 255
                            d = lossless_d[next_lossless_d]
                            next_lossless_d += 1
                        else
                            d = d + aeb * (dCodes[i,j,t] - 127) / (2^precision)
                        end

                        if rCodes[i,j,t] == 255
                            r = lossless_r[next_lossless_r]
                            next_lossless_r += 1
                        else
                            r = r + aeb * (rCodes[i,j,t] - 127) / (2^precision)
                        end

                        if sCodes[i,j,t] == 255
                            s = lossless_s[next_lossless_s]
                            next_lossless_s += 1
                        else
                            s = s + sqrt(2) * aeb * (sCodes[i,j,t] - 127) / (2^precision)
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
                            r = 0.0
                        end

                        setTensor(tf, i, j, t, recomposeTensor(d, r, s, θ))

                    end # end if precision >= 8 (then else for the main reconstruction)

                end # end if not all of the base codes are 0

            end # end for
        end # end for 
    end # end for

    augmentSplit = time()

    # tf2 = loadTensorField2dFromFolder("../output/test", dims_tuple)

    # for t in 1:dims[3]
    #     for j in 1:dims[2]
    #         for i in 1:dims[1]
    #             if getTensor(tf,i,j,t) != getTensor(tf2,i,j,t)
    #                 println("mismatch at $((i,j,t))")
    #                 println(getTensor(tf,i,j,t))
    #                 println(decomposeTensor(getTensor(tf,i,j,t)))
    #                 println("---")
    #                 println(getTensor(tf2,i,j,t))
    #                 println(decomposeTensor(getTensor(tf2,i,j,t)))
    #             end
    #         end
    #     end
    # end

    # Save to file
    saveTensorField64("$output/$decompress_folder", tf)

    saveSplit = time()

    remove("$output/row_1_col_1.dat")
    remove("$output/row_1_col_2.dat")
    remove("$output/row_2_col_1.dat")
    remove("$output/row_2_col_2.dat")
    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_1.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/vals.bytes")

    endTime = time()

    return [zstdSplit - startTime, tarSplit - zstdSplit, readSplit - tarSplit, baseDecompressorSplit - readSplit, read2Split - baseDecompressorSplit, augmentSplit - read2Split, saveSplit - augmentSplit, endTime - saveSplit ]
end

function decompress2dSymmetric(compressed_file, decompress_folder, bits, output = "../output", baseCompressor = "sz3")
    startTime = time()
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $output/$compressed_file.tar.zst`)
    zstdSplit = time()
    run(`tar xvf $output/$compressed_file.tar`)
    tarSplit = time()

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    codeBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    codeBytes = read(vals_file, codeBytesLength)
    fullLosslessBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    fullLosslessBytes = read(vals_file, fullLosslessBytesLength)
    losslessValues = reinterpret(Float64, read(vals_file))
    close(vals_file)

    codesBase = huffmanDecode(codeBytes)
    dims_tuple::Tuple{Int64,Int64,Int64} = (dims[1],dims[2],dims[3])
    if length(codesBase) == 0
        codes = zeros(Int64, dims_tuple)
    else
        codes = reshape(codesBase,dims_tuple)
    end

    fullLosslessBase = huffmanDecode(fullLosslessBytes)
    if length(fullLosslessBase) == 0
        fullLossless = zeros(Int64, dims_tuple)
    else
        fullLossless = reshape(fullLosslessBase, dims_tuple)
    end

    readSplit = time()

    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -f -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3-master/build/bin/sz3 -f -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3-master/build/bin/sz3 -f -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/$decompress_folder/row_1_col_1.dat --pwe $bound`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/$decompress_folder/row_1_col_2.dat --pwe $bound`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/$decompress_folder/row_2_col_2.dat --pwe $bound`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end

    baseDecompressorSplit = time()

    tf = loadTensorField2dSymmetricFromFolder("$output/$decompress_folder", dims_tuple)

    read2Split = time()

    # adjust

    next_lossless = 1
    for t in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]
                if fullLossless[i,j,t] == 1
                    nextTensor = SVector{3,Float64}(losslessValues[next_lossless], losslessValues[next_lossless+1], losslessValues[next_lossless+2])
                    setTensor(tf, i, j, t, nextTensor)
                    next_lossless += 3
                elseif codes[i,j,t] == 2^bits-1
                    intermediateTensor = getTensor(tf, i, j, t)
                    trace = (intermediateTensor[1]+intermediateTensor[3])/2
                    nextTensor = SVector{3,Float64}(losslessValues[next_lossless]+trace, losslessValues[next_lossless+1], -losslessValues[next_lossless]+trace)
                    setTensor(tf, i, j, t, nextTensor)
                    next_lossless += 2
                elseif codes[i,j,t] != 0
                    trace, r, θ = decomposeTensorSymmetric( getTensor( tf, i, j, t ) )
                    θ += 2pi/(2^bits-1)*codes[i,j,t]
                    setTensor(tf, i, j, t, recomposeTensorSymmetric( trace, r, θ ))
                end
            end
        end
    end

    augmentSplit = time()

    saveTensorFieldSymmetric64("$output/$decompress_folder", tf)

    saveSplit = time()

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")
    remove("$output/vals.bytes")

    endTime = time()

    return [zstdSplit - startTime, tarSplit - zstdSplit, readSplit - tarSplit, baseDecompressorSplit - readSplit, read2Split - baseDecompressorSplit, augmentSplit - read2Split, saveSplit - augmentSplit, endTime - saveSplit ]

end

end