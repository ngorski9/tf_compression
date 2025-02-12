module compress

using LinearAlgebra
using DataStructures
using StaticArrays

using ..tensorField
using ..decompress
using ..huffman
using ..utils
using ..conicUtils
using ..cellTopology

export compress2d
export compress2dNaive
export compress2dSymmetric
export compress2dSymmetricNaive
export compress2dSymmetricNaiveWithMask

function compress2dNaive(containing_folder, dims, output_file, relative_error_bound, output = "../output", baseCompressor="sz3", verbose=false)
    tf = loadTensorField2dFromFolder(containing_folder, dims)

    # prepare derived attributes for compression

    min_entry, max_entry = getMinAndMax(tf)

    aeb = relative_error_bound * (max_entry - min_entry)

    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_1.dat -z $output/row_1_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_2.dat -z $output/row_1_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_2_col_1.dat -z $output/row_2_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_2_col_2.dat -z $output/row_2_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    elseif baseCompressor == "mgard"
        run(`../MGARD/build/bin/mgard-cpu -z -t d -i $containing_folder/row_1_col_1.dat -c $output/row_1_col_1.cmp -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $aeb -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -z -t d -i $containing_folder/row_1_col_2.dat -c $output/row_1_col_2.cmp -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $aeb -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -z -t d -i $containing_folder/row_2_col_1.dat -c $output/row_2_col_1.cmp -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $aeb -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -z -t d -i $containing_folder/row_2_col_2.dat -c $output/row_2_col_2.cmp -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $aeb -s $smoothness`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_1_col_1.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_1.cmp --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_1_col_2.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_2.cmp --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_2_col_1.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_2_col_1.cmp --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_2_col_2.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_2_col_2.cmp --pwe $aeb`)
    elseif baseCompressor == "zfp"
        run(`../zfp/build/bin/zfp -d -i $containing_folder/row_1_col_1.dat -z $output/row_1_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $aeb`)
        run(`../zfp/build/bin/zfp -d -i $containing_folder/row_1_col_2.dat -z $output/row_1_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $aeb`)
        run(`../zfp/build/bin/zfp -d -i $containing_folder/row_2_col_1.dat -z $output/row_2_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $aeb`)
        run(`../zfp/build/bin/zfp -d -i $containing_folder/row_2_col_2.dat -z $output/row_2_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $aeb`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])
    write(vals_file, aeb)
    close(vals_file)

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.zst")

    run(`tar cvf $output_file.tar row_1_col_1.cmp row_1_col_2.cmp row_2_col_1.cmp row_2_col_2.cmp vals.bytes`)
    run(`zstd $output_file.tar`)

    # removeIfExists("$output_file.tar")

    cd(cwd)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_1.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/vals.bytes")

    return [ 0.0 ]

end

function compress2dSymmetricNaive(containing_folder, dims, output_file, relative_error_bound, output = "../output", baseCompressor = "sz3")
    tf = loadTensorField2dSymmetricFromFolder(containing_folder, dims)

    # prepare derived attributes for compression

    min_entry, max_entry = getMinAndMax(tf)

    aeb = relative_error_bound * (max_entry - min_entry)

    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_1.dat -z $output/row_1_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_2.dat -z $output/row_1_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_2_col_2.dat -z $output/row_2_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_1_col_1.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_1.cmp --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_1_col_2.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_2.cmp --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_2_col_2.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_2_col_2.cmp --pwe $aeb`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])        
    write(vals_file, relative_error_bound)
    close(vals_file)

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.zst")

    run(`tar cvf $output_file.tar row_1_col_1.cmp row_1_col_2.cmp row_2_col_2.cmp vals.bytes`)
    run(`zstd $output_file.tar`)

    # removeIfExists("$output_file.tar")

    cd(cwd)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/vals.bytes")

    return [0.0]
end

function compress2dSymmetricNaiveWithMask(containing_folder, dims, output_file, relative_error_bound, output = "../output", baseCompressor = "sz3")
    tf = loadTensorField2dSymmetricFromFolder(containing_folder, dims)

    # prepare derived attributes for compression

    min_entry = tf.entries[1,1,1]
    max_entry = tf.entries[1,1,1]

    aeb = relative_error_bound * (max_entry - min_entry)
    mask = ones(Int64, dims)

    for k in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]

                for t in 1:3
                    if tf.entries[t,i,j,k] < min_entry
                        min_entry = tf.entries[t,i,j,k]
                    elseif tf.entries[t,i,j,k] > max_entry
                        max_entry = tf.entries[t,i,j,k]
                    end
                end

                if tf.entries[1,i,j,k] == 0.0 && tf.entries[2,i,j,k] == 0.0 && tf.entries[3,i,j,k] == 0.0
                    mask[i,j,k] = 0
                end

            end
        end
    end

    aeb = relative_error_bound * (max_entry - min_entry)

    maskBytes = huffmanEncode(vec(mask))

    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_1.dat -z $output/row_1_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_2.dat -z $output/row_1_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_2_col_2.dat -z $output/row_2_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_1_col_1.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_1.cmp --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_1_col_2.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_2.cmp --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $containing_folder/row_2_col_2.dat -c --ftype 64 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_2_col_2.cmp --pwe $aeb`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
    end

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])        
    write(vals_file, relative_error_bound)
    write(vals_file, maskBytes)
    close(vals_file)

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.zst")

    run(`tar cvf $output_file.tar row_1_col_1.cmp row_1_col_2.cmp row_2_col_2.cmp vals.bytes`)
    run(`zstd $output_file.tar`)

    cd(cwd)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/vals.bytes")

    return [0.0]
end

function compress2d(containing_folder, dims, output_file, relative_error_bound, output="../output", verbose=false, eigenvalue=true, eigenvector=true, minCrossing = 0.01, baseCompressor = "sz3", parameter=1.0)
    startTime = time()
    tf = loadTensorField2dFromFolder(containing_folder, dims)
    
    # prepare derived attributes for compression

    max_entry = -Inf
    min_entry = Inf

    d_ground = ones(Float64, dims)
    r_ground = ones(Float64, dims)
    s_ground = ones(Float64, dims)
    θ_ground = ones(Float64, dims)

    for t in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]

                matrix = getTensor(tf, i, j, t)
                d,r,s,θ = decomposeTensor(matrix)

                d_ground[i,j,t] = d
                r_ground[i,j,t] = r
                s_ground[i,j,t] = s
                θ_ground[i,j,t] = θ

                max_entry = max(max_entry, maximum(matrix))
                min_entry = min(min_entry, minimum(matrix))

            end
        end
    end

    aeb = relative_error_bound * (max_entry - min_entry)

    setup1Split = time()

    saveTensorField32(output, tf, "_g")

    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -f -i $output/row_1_col_1_g.dat -z $output/row_1_col_1.cmp -o $output/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $(parameter*aeb)`)
        run(`../SZ3-master/build/bin/sz3 -f -i $output/row_1_col_2_g.dat -z $output/row_1_col_2.cmp -o $output/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $(parameter*aeb)`)
        run(`../SZ3-master/build/bin/sz3 -f -i $output/row_2_col_1_g.dat -z $output/row_2_col_1.cmp -o $output/row_2_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $(parameter*aeb)`)
        run(`../SZ3-master/build/bin/sz3 -f -i $output/row_2_col_2_g.dat -z $output/row_2_col_2.cmp -o $output/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $(parameter*aeb)`)
    elseif baseCompressor == "mgard"
        run(`../MGARD/build/bin/mgard-cpu -z -t s -i $output/row_1_col_1_g.dat -c $output/row_1_col_1.cmp -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -z -t s -i $output/row_1_col_2_g.dat -c $output/row_1_col_2.cmp -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -z -t s -i $output/row_2_col_1_g.dat -c $output/row_2_col_1.cmp -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -z -t s -i $output/row_2_col_2_g.dat -c $output/row_2_col_2.cmp -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)

        run(`../MGARD/build/bin/mgard-cpu -x -t s -c $output/row_1_col_1.cmp -d $output/row_1_col_1.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t s -c $output/row_1_col_2.cmp -d $output/row_1_col_2.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t s -c $output/row_2_col_1.cmp -d $output/row_2_col_1.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t s -c $output/row_2_col_2.cmp -d $output/row_2_col_2.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $(parameter*aeb) -s $smoothness`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_1_g.dat -c --ftype 32 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_1.cmp --pwe $(parameter*aeb)`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_2_g.dat -c --ftype 32 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_2.cmp --pwe $(parameter*aeb)`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_1_g.dat -c --ftype 32 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_2_col_1.cmp --pwe $(parameter*aeb)`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_2_g.dat -c --ftype 32 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_2_col_2.cmp --pwe $(parameter*aeb)`)

        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_1_col_1.dat --pwe $(parameter*aeb)`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_1_col_2.dat --pwe $(parameter*aeb)`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_1.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_2_col_1.dat --pwe $(parameter*aeb)`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_2_col_2.dat --pwe $(parameter*aeb)`)
    end

    baseCompressorSplit = time()

    tf2 = loadTensorField2dFromFolder(output, dims)

    d_intermediate = zeros(Float64, dims)
    r_intermediate = zeros(Float64, dims)
    s_intermediate = zeros(Float64, dims)
    θ_intermediate = zeros(Float64, dims)

    # Final is perhaps a misnomer.
    # These are the values after quantization but before swapping.
    d_final = zeros(Float64, dims)
    r_final = zeros(Float64, dims)
    s_final = zeros(Float64, dims)
    θ_final = zeros(Float64, dims)

    d_codes = 127*ones(Int64, dims)
    r_codes = 127*ones(Int64, dims)
    s_codes = 127*ones(Int64, dims)
    θ_codes = zeros(UInt8, dims)

    type_codes = zeros(UInt8, dims)
    precisions = zeros(UInt8, dims)
    angles_mandatory = zeros(Bool, dims)
    θ_quantized = Inf*ones(Float64, dims)
    sfix_codes = zeros(UInt8, dims)
    eigenvector_special_cases = zeros(UInt8, dims)
    has_ellipses = zeros(Bool, (dims[1]-1, dims[2]-1, dims[3], 2))

    function quantize_angle(i,j,t)
        θ_dif = θ_ground[i,j,t] - θ_intermediate[i,j,t]
        if θ_dif < 0
            θ_dif += 2pi
        end

        θ_code = UInt8(round( θ_dif * (2^6-1) / 2pi ))
        if θ_code == 2^6-1
            θ_code = 0
        end
        θ_codes[i,j,t] = θ_code
        θ_quantized[i,j,t] = θ_intermediate[i,j,t] + 2pi/(2^6-1)*θ_code
    end

    # nested function to raise the precision of a 
    function raise_precision(i,j,t)
        if θ_final[i,j,t] == θ_quantized[i,j,t]

            if precisions[i,j,t] >= 7
                precisions[i,j,t] = 8
                d_final[i,j,t] = d_ground[i,j,t]
                r_final[i,j,t] = r_ground[i,j,t]
                s_final[i,j,t] = s_ground[i,j,t]
                θ_final[i,j,t] = θ_ground[i,j,t]
                setTensor(tf2, i,j,t, getTensor(tf,i,j,t))
                # if (i,j,t) == (47,30,1)
                #     println("set 1")
                # end

            else
                precisions[i,j,t] += 1
                if eigenvalue
                    d_code = Int64(round((d_ground[i,j,t]-d_intermediate[i,j,t])*(2^precisions[i,j,t])/aeb))

                    if d_code < -127 || d_code > 127
                        d_final[i,j,t] = d_ground[i,j,t]
                        d_codes[i,j,t] = 255                        
                    else
                        d_final[i,j,t] = d_intermediate[i,j,t] + aeb * d_code / (2^precisions[i,j,t])
                        d_codes[i,j,t] = d_code + 127                        
                    end
                end

                r_code = Int64(round((r_ground[i,j,t]-r_intermediate[i,j,t])*(2^precisions[i,j,t])/aeb))
                s_code = Int64(round((s_ground[i,j,t]-s_intermediate[i,j,t])*(2^precisions[i,j,t])/(sqrt(2)*aeb)))

                if r_code < -127 || r_code > 127
                    r_final[i,j,t] = r_ground[i,j,t]
                    r_codes[i,j,t] = 255                    
                else
                    r_final[i,j,t] = r_intermediate[i,j,t] + aeb * r_code / (2^precisions[i,j,t])
                    r_codes[i,j,t] = r_code + 127                    
                end

                if s_code < -127 || s_code > 127
                    s_final[i,j,t] = s_ground[i,j,t]
                    s_codes[i,j,t] = 255
                else
                    s_final[i,j,t] = s_intermediate[i,j,t] + sqrt(2) * aeb * s_code / (2^precisions[i,j,t])
                    s_codes[i,j,t] = s_code + 127                    
                end

                if !angles_mandatory[i,j,t]
                    θ_final[i,j,t] = θ_intermediate[i,j,t]
                end
            end
        else
            if θ_quantized[i,j,t] == Inf
                quantize_angle(i,j,t)
            end

            θ_final[i,j,t] = θ_quantized[i,j,t]
        end
    end # end function raise_precision

    # internal function 2
    function processPoint(coords)

        pending = true

        while pending

            d_swap = d_final[coords...]
            r_swap = r_final[coords...]
            s_swap = s_final[coords...]
            θ_swap = θ_final[coords...]

            s_fix = 0
            
            if precisions[coords...] != 0
                # account for s possibly being negative
                if s_swap < -(sqrt(2)-1)*aeb
                    s_swap += sqrt(2)*aeb
                elseif s_swap < 0
                    s_swap += (sqrt(2)-1)*aeb
                end
            end

            d_sign_swap = false
            r_sign_swap = false
            d_largest_swap = false
            r_over_s_swap = false
            degenerateCase = false

            # Check whether any modifications need to take place at all.
            modifications = false

            if eigenvector
                eigenvectorRecon = classifyTensorEigenvector(r_swap, s_swap)
                eigenvectorGround = classifyTensorEigenvector(r_ground[coords...], s_ground[coords...])
            else
                eigenvectorRecon = 0
                eigenvectorGround = 0
            end

            if eigenvalue
                eigenvalueRecon = classifyTensorEigenvalue(d_swap, r_swap, s_swap)
                eigenvalueGround = classifyTensorEigenvalue(d_ground[coords...], r_ground[coords...], s_ground[coords...])
            else
                eigenvalueRecon = 0
                eigenvalueGround = 0
            end

            if !( eigenvectorRecon == eigenvectorGround && eigenvalueRecon == eigenvalueGround && maximum(abs.(getTensor(tf,coords...)-getTensor(tf2,coords...))) <= aeb)

                # check whether any of d, r, or s are equal such that swapping wouldn't work. If so, raise precision.
                degenerateCase = ((eigenvalueGround == POSITIVE_SCALING || eigenvalueGround == NEGATIVE_SCALING) && (abs(d_swap) == abs(r_swap) || abs(d_swap) == s_swap)) || abs(r_swap) == s_swap

                if !degenerateCase
                    modifications = true

                    if precisions[coords...] != 0
                        # account for s possibly being negative
                        if s_swap < -(sqrt(2)-1)*aeb
                            s_swap += sqrt(2)*aeb
                        elseif s_swap < 0
                            s_swap += (sqrt(2)-1)*aeb
                        end
                    end

                    # account for s possibly being out of range (max error is sqrt(2)aeb )
                    if s_swap - s_ground[coords...] > aeb
                        s_swap -= (sqrt(2)-1)*aeb
                        s_fix = 1
                    elseif s_swap - s_ground[coords...] < -aeb
                        s_swap += (sqrt(2)-1)*aeb
                        s_fix = 2
                    end

                    # eigenvector special cases and r_sign_swap

                    if eigenvector || ( abs(r_ground[coords...]) >= abs(d_ground[coords...]) && s_ground[coords...] >= abs(d_ground[coords...]) && abs(r_ground[coords...]) == s_ground[coords...] )
                        if eigenvectorGround == SYMMETRIC && eigenvectorRecon != eigenvectorGround
                            eigenvector_special_cases[coords...] = 1
                        else
                            r_sign_swap = ( r_ground[coords...] > 0 ) ⊻ ( r_swap > 0 )
                            if (eigenvectorGround == PI_BY_4 || eigenvectorGround == MINUS_PI_BY_4) && eigenvectorRecon != eigenvectorGround
                                eigenvector_special_cases[coords...] = 2
                            end
                        end
                    elseif abs(r_ground[coords...]) == abs(s_ground[coords...]) && (abs(r_ground[coords...]) > abs(d_ground[coords...]) || abs(s_ground[coords...]) > abs(d_ground[coords...]))
                        eigenvector_special_cases[coords...] = 1
                    end

                    if r_sign_swap
                        if r_swap < 0
                            r_swap += aeb
                        else
                            r_swap -= aeb
                        end
                    end

                    # d sign swap

                    if eigenvalue
                        d_sign_swap = ( abs(d_ground[coords...]) > abs(r_ground[coords...]) && abs(d_ground[coords...]) > s_ground[coords...] ) && ( (d_ground[coords...] > 0) ⊻ (d_swap > 0) )

                        if d_sign_swap
                            if d_swap < 0
                                d_swap += aeb
                            else
                                d_swap -= aeb
                            end
                        end
                    end

                    # magnitude swap codes

                    if eigenvalue
                        d_largest_swap = ( abs(d_ground[coords...]) >= abs(r_ground[coords...]) && abs(d_ground[coords...]) >= s_ground[coords...] ) ⊻ (abs(d_swap) > abs(r_swap) && abs(d_swap) > s_swap)

                        if d_largest_swap
                            # swap d with the larger of r or s
                            if abs(r_swap) > s_swap
                                sign_d = (d_swap >= 0) ? 1 : -1
                                sign_r = (r_swap >= 0) ? 1 : -1
                                temp = d_swap
                                d_swap = sign_d * abs(r_swap)
                                r_swap = sign_r * abs(temp)
                            else
                                sign_d = (d_swap >= 0) ? 1 : -1
                                temp = d_swap
                                d_swap = sign_d * s_swap
                                s_swap = abs(temp)
                            end
                        end

                    end

                    if eigenvector || (abs(r_ground[coords...]) >= abs(d_ground[coords...]) || s_ground[coords...] >= abs(d_ground[coords...]))
                        if eigenvector_special_cases[coords...] == 0
                            r_over_s_swap = (abs(r_ground[coords...]) >= s_ground[coords...]) ⊻ (abs(r_swap) >= s_swap)

                            if r_over_s_swap
                                sign_r = (r_swap >= 0) ? 1 : -1
                                temp = r_swap
                                r_swap = sign_r * s_swap
                                s_swap = abs(temp)
                            end
                        elseif eigenvector_special_cases[coords...] == 1
                            sign_r = (r_swap >= 0) ? 1 : -1
                            r_swap = sign_r * s_swap
                        else
                            r_swap = 0.0
                        end
                    end
                end # end if not degenerate case
            end # end if the default classifications do not match

            # check whether case is degenerate, or whether no changes were made, or whether changes were made and those changes are valid.
            # then, tighten or terminate accordingly.
            if degenerateCase
                raise_precision(coords...)
            elseif precisions[coords...] >= 8 || (!modifications && precisions[coords...] == 0 && θ_final[coords...] == θ_intermediate[coords...])
                pending = false
            else
                reconstructed = recomposeTensor(d_swap, r_swap, s_swap, θ_swap)
                d2, r2, s2, θ2 = decomposeTensor(reconstructed)

                if maximum(abs.(reconstructed - getTensor(tf, coords...))) <= aeb &&
                    (!eigenvector || classifyTensorEigenvector(r2, s2) == classifyTensorEigenvector(r_ground[coords...], s_ground[coords...])) &&
                    (!eigenvalue || classifyTensorEigenvalue(d2, r2, s2) == classifyTensorEigenvalue(d_ground[coords...], r_ground[coords...], s_ground[coords...]))

                    pending = false
                    sfix_codes[coords...] = s_fix
                    type_codes[coords...] = (d_sign_swap << 3) | (r_sign_swap << 2) | (d_largest_swap << 1) | (r_over_s_swap)

                    setTensor(tf2, coords..., reconstructed)
                    # if coords == (47,30,1)
                    #     println("set 2")
                    # end
                else
                    raise_precision(coords...)
                    eigenvector_special_cases[coords...] = 0 # reset special case as raising precision could affect this.
                end
            end

        end # end while pending

    end # end function process_point

    println("initial processing...")

    for t in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]

                matrix = getTensor(tf2, i, j, t)
                d,r,s,θ = decomposeTensor(matrix)

                d_intermediate[i,j,t] = d
                d_final[i,j,t] = d
                r_intermediate[i,j,t] = r
                r_final[i,j,t] = r
                s_intermediate[i,j,t] = s
                s_final[i,j,t] = s
                θ_intermediate[i,j,t] = θ
                θ_final[i,j,t] = θ

            end
        end
    end

    println("correcting errors...")

    stack::Array{Tuple{Int64,Int64,Bool,Bool}} = Array{Tuple{Int64,Int64,Bool,Bool}}(undef, 0)

    setup2Split = time()
    numCellsProcessed = 0.0
    numCellsModified = 0.0
    individualPointsTime = 0.0
    circularPointsTime = 0.0
    cellTopologyTime = 0.0
    queueTime = 0.0

    if aeb != 0.0

        for t in 1:dims[3]
            for j in 1:dims[2]-1
                for i in 1:dims[1]-1
                    for k in 0:1

                        pushSplit1 = time()
                        push!(stack, (i,j,Bool(k),true) )
                        pushSplit2 = time()

                        queueTime += pushSplit2 - pushSplit1

                        while length(stack) > 0

                            individualSplit1 = time()

                            numCellsProcessed += 1
                            x,y,top,processNewVertices = pop!(stack)

                            # bottom left, bottom right, top left (corresponds to coords of bottom in order) then top right
                            vertices_modified = [false,false,false,false]

                            vertexCoords = getCellVertexCoords(x,y,t,top)

                            if processNewVertices

                                # identify then process new vertices
                                if top
                                    newVertices = [3]
                                else
                                    if x == 1
                                        if y == 1
                                            newVertices = [1,2,3]
                                        else
                                            newVertices = [3]
                                        end
                                    elseif y == 1
                                        newVertices = [2]
                                    else
                                        newVertices::Array{Int64} = Array{Int64}(undef,0)
                                    end
                                end


                                # Single vertex: swap values into place.
                                for v in newVertices
                                    if abs(d_ground[vertexCoords[v]...]) == abs(r_ground[vertexCoords[v]...]) || abs(d_ground[vertexCoords[v]...]) == s_ground[vertexCoords[v]...] || abs(r_ground[vertexCoords[v]...]) == s_ground[vertexCoords[v]...]
                                        precisions[vertexCoords[v]...] = 8
                                        setTensor(tf2, vertexCoords[v]..., getTensor(tf, vertexCoords[v]...))
                                        θ_final[vertexCoords[v]...] = θ_ground[vertexCoords[v]...]
                                    else
                                        processPoint(vertexCoords[v])
                                    end
                                    processPoint(vertexCoords[v])
                                end

                            end

                            individualSplit2 = time()
                            individualPointsTime += individualSplit2 - individualSplit1

                            # Process individual cells to check circular points.
                            if eigenvector
                                groundCircularPointType = getCircularPointType(tf, x, y, t, top)
                                if groundCircularPointType != getCircularPointType(tf2, x, y, t, top)

                                    θe1 = abs(θ_final[vertexCoords[1]...] - θ_ground[vertexCoords[1]...])
                                    θe2 = abs(θ_final[vertexCoords[2]...] - θ_ground[vertexCoords[2]...])
                                    θe3 = abs(θ_final[vertexCoords[3]...] - θ_ground[vertexCoords[3]...])

                                    if θe1 > pi
                                        θe1 = abs(θe1-2pi)
                                    end

                                    if θe2 > pi
                                        θe2 = abs(θe2-2pi)
                                    end

                                    if θe3 > pi
                                        θe3 = abs(θe3-2pi)
                                    end

                                    θe = [θe1, θe2, θe3]
                                    ll = [θe1==0.0, θe2==0.0, θe3==0.0]

                                    while groundCircularPointType != getCircularPointType(tf2, x, y, t, top)
                                        if ll[1] && ll[2] && ll[3]

                                            precisions[vertexCoords[1]...] = 8
                                            precisions[vertexCoords[2]...] = 8
                                            precisions[vertexCoords[3]...] = 8

                                            setTensor(tf2, vertexCoords[1]..., getTensor(tf, vertexCoords[1]...))
                                            setTensor(tf2, vertexCoords[2]..., getTensor(tf, vertexCoords[2]...))
                                            setTensor(tf2, vertexCoords[3]..., getTensor(tf, vertexCoords[3]...))

                                            θ_final[vertexCoords[1]...] = θ_ground[vertexCoords[1]...]
                                            θ_final[vertexCoords[2]...] = θ_ground[vertexCoords[2]...]
                                            θ_final[vertexCoords[3]...] = θ_ground[vertexCoords[3]...]

                                            if top
                                                vertices_modified[2] = true
                                                vertices_modified[3] = true
                                                vertices_modified[4] = true
                                            else
                                                vertices_modified[1] = true
                                                vertices_modified[2] = true
                                                vertices_modified[3] = true
                                            end

                                        else

                                            changeTensor = findmax(θe)[2]

                                            if angles_mandatory[vertexCoords[changeTensor]...]
                                                θ_quantized[vertexCoords[changeTensor]...] = θ_ground[vertexCoords[changeTensor]...]
                                                θ_final[vertexCoords[changeTensor]...] = θ_ground[vertexCoords[changeTensor]...]
                                                θ_codes[vertexCoords[changeTensor]...] = 2^6-1
                                                ll[changeTensor] = true
                                                θe[changeTensor] = 0.0
                                            else
                                                angles_mandatory[vertexCoords[changeTensor]...] = true
                                                if θ_quantized[vertexCoords[changeTensor]...] == Inf
                                                    quantize_angle(vertexCoords[changeTensor]...)
                                                end
                                            end
                                            
                                            if θ_codes[vertexCoords[changeTensor]...] != 0 && precisions[vertexCoords[changeTensor]...] < 8

                                                θ_final[vertexCoords[changeTensor]...] = θ_quantized[vertexCoords[changeTensor]...]

                                                setTensor(tf2, vertexCoords[changeTensor]..., recomposeTensor(d_final[vertexCoords[changeTensor]...],
                                                                                                            r_final[vertexCoords[changeTensor]...],
                                                                                                            s_final[vertexCoords[changeTensor]...],
                                                                                                            θ_final[vertexCoords[changeTensor]...]))
                                                                                                            
                                                processPoint(vertexCoords[changeTensor])

                                                θe[changeTensor] = abs(θ_final[vertexCoords[changeTensor]...] - θ_ground[vertexCoords[changeTensor]...])

                                                if θe[changeTensor] > pi
                                                    θe[changeTensor] = abs(θe[changeTensor]-2pi)
                                                end

                                                if θe[changeTensor] == 0.0
                                                    ll[changeTensor] = true
                                                end

                                                if top
                                                    if changeTensor == 1
                                                        vertices_modified[3] = true
                                                    elseif changeTensor == 2
                                                        vertices_modified[2] = true
                                                    else
                                                        vertices_modified[4] = true
                                                    end
                                                else
                                                    vertices_modified[changeTensor] = true
                                                end
                                            end

                                        end

                                    end

                                end
                            end

                            circularPointsSplit = time()
                            circularPointsTime += circularPointsSplit - individualSplit2

                            # process cell topology
                            if eigenvalue
                                # we need to name gt something different in the 2 cases for type stability purposes.
                                gtval = tensorField.classifyCellEigenvalue(tf, x, y, t, top, eigenvector)
                                rtval = tensorField.classifyCellEigenvalue(tf2, x, y, t, top, eigenvector)
                                
                                while !( gtval.vertexTypesEigenvalue == rtval.vertexTypesEigenvalue && gtval.DPArray == rtval.DPArray && gtval.DNArray == rtval.DNArray &&
                                        gtval.RPArray == rtval.RPArray && gtval.RNArray == rtval.RNArray && (!eigenvector || (gtval.vertexTypesEigenvector == rtval.vertexTypesEigenvector &&
                                        gtval.RPArrayVec == rtval.RPArrayVec && gtval.RNArrayVec == rtval.RNArrayVec )))

                                    if precisions[vertexCoords[1]...] != 8 && ((gtval.vertexTypesEigenvalue[1] != rtval.vertexTypesEigenvalue[1] && gtval.vertexTypesEigenvalue[1] in [DPRP,DPRN,DNRP,DNRN,Z]) ||
                                        (eigenvector && gtval.vertexTypesEigenvector[1] != rtval.vertexTypesEigenvector[1] && gtval.vertexTypesEigenvector[1] in [DegenRP, SYM, DegenRN]) ||
                                        gtval.hits_corners[1])

                                        precisions[vertexCoords[1]...] = 8
                                        setTensor(tf2, vertexCoords[1]..., getTensor(tf, vertexCoords[1]...))
                                        θ_final[vertexCoords[1]...] = θ_ground[vertexCoords[1]...]
                                        if top
                                            vertices_modified[3] = true
                                        else
                                            vertices_modified[1] = true
                                        end

                                    elseif precisions[vertexCoords[2]...] != 8 && ((gtval.vertexTypesEigenvalue[2] != rtval.vertexTypesEigenvalue[2] && gtval.vertexTypesEigenvalue[2] in [DPRP,DPRN,DNRP,DNRN,Z]) ||
                                        (eigenvector && gtval.vertexTypesEigenvector[2] != rtval.vertexTypesEigenvector[2] && gtval.vertexTypesEigenvector[2] in [DegenRP, SYM, DegenRN]) ||
                                        gtval.hits_corners[2])

                                        precisions[vertexCoords[2]...] = 8
                                        setTensor(tf2, vertexCoords[2]..., getTensor(tf, vertexCoords[2]...))
                                        θ_final[vertexCoords[2]...] = θ_ground[vertexCoords[2]...]
                                        vertices_modified[2] = true

                                    elseif precisions[vertexCoords[3]...] != 8 && ((gtval.vertexTypesEigenvalue[3] != rtval.vertexTypesEigenvalue[3] && gtval.vertexTypesEigenvalue[3] in [DPRP,DPRN,DNRP,DNRN,Z]) ||
                                    (eigenvector && gtval.vertexTypesEigenvector[3] != rtval.vertexTypesEigenvector[3] && gtval.vertexTypesEigenvector[3] in [DegenRP, SYM, DegenRN]) ||
                                    gtval.hits_corners[3])

                                        precisions[vertexCoords[3]...] = 8
                                        setTensor(tf2, vertexCoords[3]..., getTensor(tf, vertexCoords[3]...))
                                        θ_final[vertexCoords[3]...] = θ_ground[vertexCoords[3]...]

                                        if top
                                            vertices_modified[4] = true
                                        else
                                            vertices_modified[3] = true
                                        end

                                    else
                                        raise_precision(vertexCoords[1]...)
                                        processPoint(vertexCoords[1])
            
                                        raise_precision(vertexCoords[2]...)
                                        processPoint(vertexCoords[2])
            
                                        raise_precision(vertexCoords[3]...)
                                        processPoint(vertexCoords[3])
            
                                        if top
                                            vertices_modified[2] = true
                                            vertices_modified[3] = true
                                            vertices_modified[4] = true
                                        else
                                            vertices_modified[1] = true
                                            vertices_modified[2] = true
                                            vertices_modified[3] = true
                                        end
                                    end

                                    rtval = tensorField.classifyCellEigenvalue(tf2, x, y, t, top, eigenvector)           
                                end

                            elseif eigenvector
                                gtvec = tensorField.classifyCellEigenvector(tf, x, y, t, top)
                                rtvec = tensorField.classifyCellEigenvector(tf2, x, y, t, top)
                                while gtvec.vertexTypes != rtvec.vertexTypes || gtvec.RPArray != rtvec.RPArray || gtvec.RNArray != rtvec.RNArray
                                    raise_precision(vertexCoords[1]...)
                                    processPoint(vertexCoords[1])
        
                                    raise_precision(vertexCoords[2]...)
                                    processPoint(vertexCoords[2])
        
                                    raise_precision(vertexCoords[3]...)
                                    processPoint(vertexCoords[3])
        
                                    if top
                                        vertices_modified[2] = true
                                        vertices_modified[3] = true
                                        vertices_modified[4] = true
                                    else
                                        vertices_modified[1] = true
                                        vertices_modified[2] = true
                                        vertices_modified[3] = true
                                    end

                                    rtvec = tensorField.classifyCellEigenvector(tf2, x, y, t, top)
                                end
                            end

                            cellTopologySplit = time()
                            cellTopologyTime += cellTopologySplit - circularPointsSplit

                            if vertices_modified[1] || vertices_modified[2] || vertices_modified[3] || vertices_modified[4]
                                push!(stack, (x,y,top,false))
                                numCellsModified += 1
                            end

                            # queue up all cells that will be affected by any current changes.
                            if vertices_modified[4] && x != dims[1] - 1 && ((y+1 < j) || (y+1 == j && x+1 <= i))
                                push!(stack, (x+1,y+1,false,false))
                            end

                            if vertices_modified[4] && ((y+1 < j) || (y+1 == j && x < i) || (y+1 == j && x == i && k == 1))
                                push!(stack, (x,y+1,true,false))
                            end

                            if (vertices_modified[3] || vertices_modified[4]) && ((y+1 < j) || (y+1 == j && x <= i))
                                push!(stack, (x,y+1,false,false))
                            end

                            if vertices_modified[3] && x != 1 && ((y+1 < j) || (y+1 == j && x-1 < i) || (y+1 == j && x-1 == i && k == 1))
                                push!(stack, (x-1,y+1,true,false))
                            end

                            if vertices_modified[3] && x != 1 && ((y+1 < j) || (y+1 == j && x-1 <= i))
                                push!(stack, (x-1,y+1,false,false))
                            end

                            if (vertices_modified[4]) && x != dims[1] - 1 && ((y < j) || (y == j && x+1 < i) || (y == j && x+i == i && k == 1))
                                push!(stack, (x+1,y,true,false))
                            end

                            if (vertices_modified[2] || vertices_modified[4]) && x != dims[1] - 1 && ((y < j) || (y == j && x+1 <= i))
                                push!(stack, (x+1,y,false,false))
                            end

                            if (vertices_modified[2] || vertices_modified[3]) && ( (y < j) || (y == j && x < i) || (y == j && x == i && k==1) )
                                push!(stack,(x,y,true,false))
                            end

                            if (vertices_modified[2] || vertices_modified[3])
                                push!(stack, (x,y,false,false))
                            end

                            if (vertices_modified[1] || vertices_modified[3]) && x != 1
                                push!(stack, (x-1,y,true,false))
                            end

                            if vertices_modified[1] && x != 1
                                push!(stack, (x-1,y,false,false))
                            end

                            if vertices_modified[2] && x != dims[1]-1 && y != 1
                                push!(stack, (x+1,y-1,true,false))
                                push!(stack, (x+1,y-1,false,false))
                            end

                            if (vertices_modified[1] || vertices_modified[2]) && y != 1
                                push!(stack, (x,y-1,true,false))
                            end

                            if vertices_modified[1] && y != 1
                                push!(stack, (x,y-1,false,false))
                            end

                            if vertices_modified[1] && x != 1 && y != 1
                                push!(stack, (x-1,y-1,true,false))
                            end

                            finalQueueSplit = time()
                            queueTime += finalQueueSplit - cellTopologySplit

                        end # end while length(stack) > 0

                    end
                end
            end
        end
    end

    processSplit = time()

    # Encoder (we will play around with concatenating the various codes later)
    base_codes = zeros(Int64, dims)
    θ_and_sfix_codes = zeros(Int64, dims)

    lossless_d::Array{Float64} = []
    lossless_r::Array{Float64} = []
    lossless_s::Array{Float64} = []
    lossless_θ::Array{Float64} = []

    lossless_A::Array{Float64} = []
    lossless_B::Array{Float64} = []
    lossless_C::Array{Float64} = []
    lossless_D::Array{Float64} = []

    println("processing codes...")

    for t in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]
                base_codes[i,j,t] = type_codes[i,j,t] | (precisions[i,j,t] << 4)

                if precisions[i,j,t] >= 8
                    θ_and_sfix_codes[i,j,t] = 0
                    eigenvector_special_cases[i,j,t] = 0
                    d_codes[i,j,t] = 127
                    r_codes[i,j,t] = 127
                    s_codes[i,j,t] = 127

                    tensor = getTensor(tf, i, j, t)
                    push!(lossless_A, tensor[1,1])
                    push!(lossless_B, tensor[1,2])
                    push!(lossless_C, tensor[2,1])
                    push!(lossless_D, tensor[2,2])
                else
                    if θ_final[i,j,t] == θ_quantized[i,j,t]
                        θ_and_sfix_codes[i,j,t] = θ_codes[i,j,t] | (sfix_codes[i,j,t] << 6)
                    else
                        θ_and_sfix_codes[i,j,t] = (sfix_codes[i,j,t] << 6)
                    end

                    if d_codes[i,j,t] == 255
                        push!(lossless_d, d_ground[i,j,t])
                    end

                    if r_codes[i,j,t] == 255
                        push!(lossless_r, r_ground[i,j,t])
                    end

                    if s_codes[i,j,t] == 255
                        push!(lossless_s, s_ground[i,j,t])
                    end

                    if θ_codes[i,j,t] == 2^6-1
                        push!(lossless_θ, θ_ground[i,j,t] )
                    end

                end

            end
        end
    end

    baseCodeBytes = huffmanEncode(vec(base_codes))
    θAndSfixBytes = huffmanEncode(vec(θ_and_sfix_codes))
    dBytes = huffmanEncode(vec(d_codes))
    rBytes = huffmanEncode(vec(r_codes))
    sBytes = huffmanEncode(vec(s_codes))
    specialCaseBytes = huffmanEncode(vec(eigenvector_special_cases))

    vals_file = open("$output/vals.bytes", "w")

    # write header
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])
    write(vals_file, aeb)

    # write various codes
    write(vals_file, length(baseCodeBytes))
    write(vals_file, baseCodeBytes)
    write(vals_file, length(θAndSfixBytes))
    write(vals_file, θAndSfixBytes)
    write(vals_file, length(dBytes))
    write(vals_file, dBytes)
    write(vals_file, length(rBytes))
    write(vals_file, rBytes)
    write(vals_file, length(sBytes))
    write(vals_file, sBytes)
    write(vals_file, length(specialCaseBytes))
    write(vals_file, specialCaseBytes)

    # write lossless values (we split them up like this)
    # so that perhaps the locality will be better for lossless compression
    # e.g. floating point numbers will have similar exponents.
    write(vals_file, length(lossless_d))
    if length(lossless_d) > 0
        write(vals_file, lossless_d)
    end
    write(vals_file, length(lossless_r))
    if length(lossless_r) > 0
        write(vals_file, lossless_r)
    end
    write(vals_file, length(lossless_s))
    if length(lossless_s) > 0
        write(vals_file, lossless_s)
    end
    write(vals_file, length(lossless_θ))
    if length(lossless_θ) > 0
        write(vals_file, lossless_θ)
    end
    write(vals_file, length(lossless_A))
    if length(lossless_A) > 0
        write(vals_file, lossless_A)
    end
    write(vals_file, length(lossless_B))
    if length(lossless_B) > 0
        write(vals_file, lossless_B)
    end
    write(vals_file, length(lossless_C))    
    if length(lossless_C) > 0
        write(vals_file, lossless_C)
    end
    write(vals_file, length(lossless_D))
    if length(lossless_D) > 0
        write(vals_file, lossless_D)
    end

    close(vals_file)

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.zst")

    writeToFileSplit = time()

    run(`tar cvf $output_file.tar row_1_col_1.cmp row_1_col_2.cmp row_2_col_1.cmp row_2_col_2.cmp vals.bytes`)
    run(`zstd $output_file.tar`)

    losslessCompressSplit = time()

    cd(cwd)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_1.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/row_1_col_1.dat")
    remove("$output/row_1_col_2.dat")
    remove("$output/row_2_col_1.dat")
    remove("$output/row_2_col_2.dat")
    remove("$output/row_1_col_1_g.dat")
    remove("$output/row_1_col_2_g.dat")
    remove("$output/row_2_col_1_g.dat")
    remove("$output/row_2_col_2_g.dat")
    remove("$output/vals.bytes")

    endTime = time()

    return [ setup1Split - startTime, baseCompressorSplit - setup1Split, setup2Split - baseCompressorSplit, individualPointsTime, circularPointsTime, 
            cellTopologyTime, queueTime, processSplit - setup2Split, numCellsModified, numCellsProcessed, 
             writeToFileSplit - processSplit, losslessCompressSplit - writeToFileSplit, endTime - losslessCompressSplit ]

end

function compress2dSymmetric(containing_folder, dims, output_file, relative_error_bound, bits, output = "../output", baseCompressor = "sz3")
    startTime = time()
    tf = loadTensorField2dSymmetricFromFolder(containing_folder, dims)

    # prepare derived attributes for compression

    max_entry = -Inf
    min_entry = Inf

    for t in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]
                matrix = getTensor(tf, i, j, t)

                max_entry = max(max_entry, maximum(matrix))
                min_entry = min(min_entry, minimum(matrix))

            end
        end
    end

    aeb = relative_error_bound * (max_entry - min_entry)

    setup1Split = time()

    saveTensorFieldSymmetric32(output, tf, "_g")

    if baseCompressor == "sz3"
        run(`../SZ3-master/build/bin/sz3 -f -i $output/row_1_col_1_g.dat -z $output/row_1_col_1.cmp -o $output/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3-master/build/bin/sz3 -f -i $output/row_1_col_2_g.dat -z $output/row_1_col_2.cmp -o $output/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3-master/build/bin/sz3 -f -i $output/row_2_col_2_g.dat -z $output/row_2_col_2.cmp -o $output/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    elseif baseCompressor == "sperr"
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_1_g.dat -c --ftype 32 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_1.cmp --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_2_g.dat -c --ftype 32 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_1_col_2.cmp --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_2_g.dat -c --ftype 32 --dims $(dims[1]) $(dims[2]) --bitstream $output/row_2_col_2.cmp --pwe $aeb`)

        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_1_col_1.dat --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_1_col_2.dat --pwe $aeb`)
        run(`../SPERR-main/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_2_col_2.dat --pwe $aeb`)        
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end

    baseCompressorSplit = time()

    tf2 = loadTensorField2dSymmetricFromFolder(output, dims)
    stack::Array{Tuple{Int64,Int64,Bool,Bool}} = Array{Tuple{Int64,Int64,Bool,Bool}}(undef, 0)

    codes = zeros(UInt64, dims)
    full_lossless = zeros(UInt64, dims)
    processed = zeros(Bool, dims)

    checkLoc = (24,12,11,false)
    checking = false

    setup2Split = time()
    numCellsProcessed = 0.0
    numCellsModified = 0.0
    individualPointsTime = 0.0
    circularPointsTime = 0.0
    queueTime = 0.0

    for t in 1:dims[3]
        for j in 1:dims[2]-1
            for i in 1:dims[1]-1
                for k in 0:1

                    queueT1 = time()
                    push!(stack, (i,j,Bool(k),true))
                    queueT2 = time()
                    queueTime += queueT2 - queueT1

                    while length(stack) > 0
                        processPointT1 = time()

                        numCellsProcessed += 1
                        x,y,top,checkNewVertices = pop!(stack)

                        # bottom left, bottom right, top left (corresponds to coords of bottom in order) then top right
                        vertices_modified = [false,false,false,false]

                        if checkNewVertices

                            if top
                                if getTensor(tf, x+1,y+1,t) == SVector{3,Float64}(0.0,0.0,0.0)
                                    full_lossless[x+1,y+1,t] = 1
                                    setTensor(tf2,x+1,y+1,t,SVector{3,Float64}(0.0,0.0,0.0))
                                    vertices_modified[4] = true
                                end
                            else
                                if x == 0
                                    if y == 0
                                        if getTensor(tf, x,y,t) == SVector{3,Float64}(0.0,0.0,0.0)
                                            full_lossless[x,y,t] = 1
                                            setTensor(tf2,x,y,t,SVector{3,Float64}(0.0,0.0,0.0))
                                            vertices_modified[1] = true
                                        end
                                    end

                                    if getTensor(tf, x,y+1,t) == SVector{3,Float64}(0.0,0.0,0.0)
                                        full_lossless[x,y+1,t] = 1
                                        setTensor(tf2,x,y+1,t,SVector{3,Float64}(0.0,0.0,0.0))
                                        vertices_modified[3] = true
                                    end
                                elseif y == 0 && getTensor(tf, x+1,y,t) == SVector{3,Float64}(0.0,0.0,0.0)
                                    full_lossless[x+1,y,t] = 1
                                    setTensor(tf2,x+1,y,t,SVector{3,Float64}(0.0,0.0,0.0))
                                    vertices_modified[2] = true
                                end
                            end

                        end

                        if checking
                            if (x,y,t,top) == checkLoc
                                println("self")
                            elseif t == checkLoc[3] && abs(x-checkLoc[1]) <= 1 && abs(y-checkLoc[2]) <= 1
                                println(("neighbor", (x,y,t,top)))
                            end
                        end

                        processPointT2 = time()
                        individualPointsTime += processPointT2 - processPointT1

                        circularPointsT1 = time()

                        crit_ground = getCriticalType(tf, x, y, t, top)
                        crit_intermediate = getCriticalType(tf2, x, y, t, top)

                        if crit_ground != crit_intermediate

                            if checking && t == checkLoc[3] && abs(x-checkLoc[1]) <= 1 && abs(y-checkLoc[2]) <= 1
                                println("edit")
                            end

                            vertexCoords = getCellVertexCoords(x,y,t,top)

                            tensor1Ground = getTensor(tf, vertexCoords[1]...)
                            tensor2Ground = getTensor(tf, vertexCoords[2]...)
                            tensor3Ground = getTensor(tf, vertexCoords[3]...)

                            numCellsModified += 1

                            tensor1Recon = getTensor(tf2, vertexCoords[1]...)
                            tensor2Recon = getTensor(tf2, vertexCoords[2]...)
                            tensor3Recon = getTensor(tf2, vertexCoords[3]...)

                            t1g, r1g, θ1g = decomposeTensorSymmetric(tensor1Ground)
                            t2g, r2g, θ2g = decomposeTensorSymmetric(tensor2Ground)
                            t3g, r3g, θ3g = decomposeTensorSymmetric(tensor3Ground)
                            t1r, r1r, θ1r = decomposeTensorSymmetric(tensor1Recon)
                            t2r, r2r, θ2r = decomposeTensorSymmetric(tensor2Recon)
                            t3r, r3r, θ3r = decomposeTensorSymmetric(tensor3Recon)

                            θe1 = abs( θ1g - θ1r )
                            θe2 = abs( θ2g - θ2r )
                            θe3 = abs( θ3g - θ3r )

                            if θe1 > pi
                                θe1 = abs(θe1 - 2pi)
                            end

                            if θe2 > pi
                                θe2 = abs(θe2 - 2pi)
                            end

                            if θe3 > pi
                                θe3 = abs(θe3 - 2pi)
                            end

                            θg = [θ1g, θ2g, θ3g]
                            θr = [θ1r, θ2r, θ3r]
                            tg = [t1g, t2g, t3g]
                            tr = [t1r, t2r, t3r]
                            rr = [r1r, r2r, r3r]
                            tensorsGround = [tensor1Ground, tensor2Ground, tensor3Ground]
                            ll = [false, false, false]

                            while crit_ground != crit_intermediate

                                if θe1 == θe2 == θe3 == 0.0
                                    if !ll[1]
                                        idx = 1
                                    elseif !ll[2]
                                        idx = 2
                                    elseif !ll[3]
                                        idx = 3
                                    else
                                        idx = 1
                                    end
                                elseif θe1 >= θe2 && θe1 >= θe3
                                    idx = 1
                                elseif θe2 >= θe1 && θe2 >= θe3
                                    idx = 2
                                else
                                    idx = 3
                                end
                                lossless = processed[vertexCoords[idx]...]

                                # that is, it hasn't been touched yet
                                if !lossless
                                    processed[vertexCoords[idx]...] = true

                                    θdif = θg[idx] - θr[idx]
                                    if θdif < 0
                                        θdif += 2pi
                                    end

                                    code = round( θdif * ( (2^bits-1) / 2pi ) )
                                    if code == 2^bits-1
                                        code = 0.0
                                    end

                                    θnew = θr[idx] + 2pi/(2^bits-1)*code
                                    tnew = recomposeTensorSymmetric(tr[idx], rr[idx], θnew)
                                    
                                    if maximum(abs.(tnew - tensorsGround[idx])) > aeb || code == 0
                                        lossless = true
                                    else
                                        codes[vertexCoords[idx]...] = code
                                        setTensor(tf2, vertexCoords[idx]..., tnew)

                                        if idx == 1
                                            θe1 = abs(θnew - θg[idx])

                                            if θe1 > pi
                                                θe1 = abs(θe1 - 2pi)
                                            end
                                        elseif idx == 2
                                            θe2 = abs(θnew - θg[idx])

                                            if θe2 > pi
                                                θe2 = abs(θe2 - 2pi)
                                            end
                                        else
                                            θe3 = abs(θnew - θg[idx])

                                            if θe3 > pi
                                                θe3 = abs(θe3 - 2pi)
                                            end
                                        end

                                        # no need to update the other values because the only other place this can go is lossless...

                                        if top
                                            if idx == 1
                                                vertices_modified[3] = true
                                            elseif idx == 2
                                                vertices_modified[2] = true
                                            elseif idx == 3
                                                vertices_modified[4] = true
                                            end
                                        else
                                            vertices_modified[idx] = true
                                        end

                                    end

                                end

                                if lossless
                                    if ll[1] && ll[2] && ll[3]                 
                                        # Storing things losslessly with the trace trick didn't do the job.
                                        # In this case, just store all three cells totally losslessly
                                        # This is very rare.

                                        codes[vertexCoords[1]...] = 0
                                        codes[vertexCoords[2]...] = 0
                                        codes[vertexCoords[3]...] = 0
                                        
                                        full_lossless[vertexCoords[1]...] = 1
                                        full_lossless[vertexCoords[2]...] = 1
                                        full_lossless[vertexCoords[3]...] = 1

                                        setTensor(tf2, vertexCoords[1]..., getTensor(tf, vertexCoords[1]...))
                                        setTensor(tf2, vertexCoords[2]..., getTensor(tf, vertexCoords[2]...))
                                        setTensor(tf2, vertexCoords[3]..., getTensor(tf, vertexCoords[3]...))

                                        if top
                                            vertices_modified[2] = true
                                            vertices_modified[3] = true
                                            vertices_modified[4] = true
                                        else
                                            vertices_modified[1] = true
                                            vertices_modified[2] = true
                                            vertices_modified[3] = true
                                        end

                                    else
                                        ll[idx] = true
                                        codes[vertexCoords[idx]...] = 2^bits-1

                                        trDif = tr[idx]-tg[idx]
                                        newTensor = SVector{3,Float64}( tensorsGround[idx][1] + trDif, tensorsGround[idx][2], tensorsGround[idx][3] + trDif )
                                        setTensor(tf2, vertexCoords[idx]..., newTensor)

                                        if idx == 1
                                            θe1 = 0.0
                                        elseif idx == 2
                                            θe2 = 0.0
                                        else
                                            θe3 = 0.0
                                        end

                                        if top
                                            if idx == 1
                                                vertices_modified[3] = true
                                            elseif idx == 2
                                                vertices_modified[2] = true
                                            elseif idx == 3
                                                vertices_modified[4] = true
                                            end
                                        else
                                            vertices_modified[idx] = true
                                        end

                                        # no need to update the other values because we're not going to touch this again.
                                    end
                                end

                                crit_intermediate = getCriticalType(tf2, x, y, t, top)

                            end

                            # requeue up any cells that must be hit after edits

                            circularPointsT2 = time()
                            circularPointsTime += circularPointsT2 - circularPointsT1

                            if vertices_modified[1] || vertices_modified[2] || vertices_modified[3] || vertices_modified[4]
                                push!(stack, (x,y,top,false))
                                numCellsModified += 1
                            end
    
                            # queue up all cells that will be affected by any current changes.
                            if vertices_modified[4] && x != dims[1] - 1 && ((y+1 < j) || (y+1 == j && x+1 <= i))
                                push!(stack, (x+1,y+1,false,false))
                            end
    
                            if vertices_modified[4] && ((y+1 < j) || (y+1 == j && x <= i))
                                push!(stack, (x,y+1,true,false))
                            end
    
                            if (vertices_modified[3] || vertices_modified[4]) && ((y+1 < j) || (y+1 == j && x <= i))
                                push!(stack, (x,y+1,false,false))
                            end
    
                            if vertices_modified[3] && x != 1 && ((y+1 < j) || (y+1 == j && x-1 <= i))
                                push!(stack, (x-1,y+1,true,false))
                                push!(stack, (x-1,y+1,false,false))
                            end
    
                            if (vertices_modified[4]) && x != dims[1] - 1 && ((y < j) || (y == j && x+1 <= i))
                                push!(stack, (x+1,y,true,false))
                            end
    
                            if (vertices_modified[2] || vertices_modified[4]) && x != dims[1] - 1 && ((y < j) || (y == j && x+1 <= i))
                                push!(stack, (x+1,y,false,false))
                            end
    
                            if (vertices_modified[2] || vertices_modified[3]) && ( (y < j) || (y == j && x < i) || (y == j && x == i && k==1) )
                                push!(stack,(x,y,true,false))
                            end
    
                            if (vertices_modified[2] || vertices_modified[3])
                                push!(stack, (x,y,false,false))
                            end
    
                            if (vertices_modified[1] || vertices_modified[3]) && x != 1
                                push!(stack, (x-1,y,true,false))
                            end
    
                            if vertices_modified[1] && x != 1
                                push!(stack, (x-1,y,false,false))
                            end
    
                            if vertices_modified[2] && x != dims[1]-1 && y != 1
                                push!(stack, (x+1,y-1,true,false))
                                push!(stack, (x+1,y-1,false,false))
                            end
    
                            if (vertices_modified[1] || vertices_modified[2]) && y != 1
                                push!(stack, (x,y-1,true,false))
                            end
    
                            if vertices_modified[1] && y != 1
                                push!(stack, (x,y-1,false,false))
                            end
    
                            if vertices_modified[1] && x != 1 && y != 1
                                push!(stack, (x-1,y-1,true,false))
                            end

                            queueSplit = time()
                            queueTime += queueSplit - circularPointsT2

                        else # end if crit_ground != crit_intermediate
                            circularPointsT2 = time()
                            circularPointsTime += circularPointsT2 - circularPointsT1
                        end

                    end # end while length(stack) > 0

                end # end for k in 0:1
            end # end for i
        end # end for j
    end # end for t

    processSplit = time()

    # for t in 1:dims[3]
    #     for j in 1:dims[2]-1
    #         for i in 1:dims[1]-1
    #             for k in 0:1

    #                 push!(stack, (i,j,Bool(k)))

    #                 while length(stack) > 0
    #                     numProcess += 1
    #                     x,y,top = pop!(stack)

    #                     crit_ground = getCircularPointType(tf, x, y, t, top)
    #                     crit_intermediate = getCircularPointType(tf2, x, y, t, top)

    #                     if crit_ground != crit_intermediate
    #                         println((i,j,t,top))
    #                         exit()
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end

    losslessValues::Vector{Float64} = []
    for t in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]
                if full_lossless[i,j,t] == 1
                    next_lossless = getTensor(tf, i, j, t)
                    push!(losslessValues, next_lossless[1])
                    push!(losslessValues, next_lossless[2])
                    push!(losslessValues, next_lossless[3])
                elseif codes[i,j,t] == 2^bits-1
                    next_lossless = getTensor(tf, i, j, t)
                    push!(losslessValues, (next_lossless[1]-next_lossless[3])/2 )
                    push!(losslessValues, next_lossless[2])
                end
            end
        end
    end

    codeBytes = huffmanEncode(vec(codes))
    full_lossless_bytes = huffmanEncode(vec(full_lossless))

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])
    write(vals_file, aeb)
    write(vals_file, length(codeBytes))
    write(vals_file, codeBytes)
    write(vals_file, length(full_lossless_bytes))
    write(vals_file, full_lossless_bytes)
    write(vals_file, losslessValues)
    close(vals_file)

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.zst")

    writeToFileSplit = time()

    run(`tar cvf $output_file.tar row_1_col_1.cmp row_1_col_2.cmp row_2_col_2.cmp vals.bytes`)
    run(`zstd $output_file.tar`)

    losslessCompressSplit = time()

    # removeIfExists("$output_file.tar")

    cd(cwd)
    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/row_1_col_1.dat")
    remove("$output/row_1_col_2.dat")
    remove("$output/row_2_col_2.dat")
    remove("$output/row_1_col_1_g.dat")
    remove("$output/row_1_col_2_g.dat")
    remove("$output/row_2_col_2_g.dat")
    remove("$output/vals.bytes")

    endTime = time()

    return [ setup1Split - startTime, baseCompressorSplit - setup1Split, setup2Split - baseCompressorSplit, individualPointsTime,
             circularPointsTime, queueTime, processSplit - setup2Split, numCellsModified, numCellsProcessed, writeToFileSplit - processSplit, 
             losslessCompressSplit - writeToFileSplit, endTime - losslessCompressSplit ]     
end

end