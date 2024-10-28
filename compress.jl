module compress

using LinearAlgebra
using DataStructures

using ..tensorField
using ..decompress
using ..huffman
using ..utils

export compress2d
export compress2dNaive
export compress2dSymmetric
export compress2dSymmetricOld
export compress2dSymmetricSimple
export compress2dSymmetricNaive

function compress2dNaive(containing_folder, dims, output_file, relative_error_bound, output = "../output")
    tf, dtype = loadTensorField2dFromFolder(containing_folder, dims)

    # prepare derived attributes for compression

    max_entry = -Inf
    min_entry = Inf

    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]
                matrix = getTensor(tf, t, i, j)

                max_entry = max(max_entry, maximum(matrix))
                min_entry = min(min_entry, minimum(matrix))

            end
        end
    end

    aeb = relative_error_bound * (max_entry - min_entry)

    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_1.dat -z $output/row_1_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_2.dat -z $output/row_1_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_2_col_1.dat -z $output/row_2_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)    
    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_2_col_2.dat -z $output/row_2_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])        
    write(vals_file, relative_error_bound)
    close(vals_file)

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.xz")

    run(`tar cvf $output_file.tar row_1_col_1.cmp row_1_col_2.cmp row_2_col_1.cmp row_2_col_2.cmp vals.bytes`)
    run(`xz -v9 $output_file.tar`)

    removeIfExists("$output_file.tar")

    cd(cwd)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_1.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/vals.bytes")

end

function compress2dSymmetricNaive(containing_folder, dims, output_file, relative_error_bound, output = "../output")
    tf, dtype = loadTensorField2dFromFolder(containing_folder, dims)

    # prepare derived attributes for compression

    max_entry = -Inf
    min_entry = Inf

    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]
                matrix = getTensor(tf, t, i, j)

                max_entry = max(max_entry, maximum(matrix))
                min_entry = min(min_entry, minimum(matrix))

            end
        end
    end

    aeb = relative_error_bound * (max_entry - min_entry)

    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_1.dat -z $output/row_1_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_2.dat -z $output/row_1_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_2_col_2.dat -z $output/row_2_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])        
    write(vals_file, relative_error_bound)
    close(vals_file)

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.xz")

    run(`tar cvf $output_file.tar row_1_col_1.cmp row_1_col_2.cmp row_2_col_2.cmp vals.bytes`)
    run(`xz -v9 $output_file.tar`)

    removeIfExists("$output_file.tar")

    cd(cwd)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/vals.bytes")
end

function compress2d(containing_folder, dims, output_file, relative_error_bound, output="../output")
    tf, dtype = loadTensorField2dFromFolder(containing_folder, dims)

    # prepare derived attributes for compression

    max_entry = -Inf
    min_entry = Inf

    d_ground = ones(Float64, dims)
    r_ground = ones(Float64, dims)
    s_ground = ones(Float64, dims)
    θ_ground = ones(Float64, dims)

    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]

                matrix = getTensor(tf, t, i, j)
                d,r,s,θ = decomposeTensor(matrix)

                d_ground[t,i,j] = d
                r_ground[t,i,j] = r
                s_ground[t,i,j] = s
                θ_ground[t,i,j] = θ

                max_entry = max(max_entry, maximum(matrix))
                min_entry = min(min_entry, minimum(matrix))

            end
        end
    end

    aeb = relative_error_bound * (max_entry - min_entry)

    saveArray("$output/row_1_col_1_g.dat", Array{Float32}(tf.entries[1,1]))
    saveArray("$output/row_1_col_2_g.dat", Array{Float32}(tf.entries[1,2]))
    saveArray("$output/row_2_col_1_g.dat", Array{Float32}(tf.entries[2,1]))
    saveArray("$output/row_2_col_2_g.dat", Array{Float32}(tf.entries[2,2]))

    run(`../SZ3-master/build/bin/sz3 -f -i $output/row_1_col_1_g.dat -z $output/row_1_col_1.cmp -o $output/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/row_1_col_2_g.dat -z $output/row_1_col_2.cmp -o $output/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/row_2_col_1_g.dat -z $output/row_2_col_1.cmp -o $output/row_2_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/row_2_col_2_g.dat -z $output/row_2_col_2.cmp -o $output/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)

    tf2, dtype2 = loadTensorField2dFromFolder(output, dims)

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

    # nested function to raise the precision of a 
    function raise_precision(t,i,j)
        if θ_final[t,i,j] == θ_quantized[t,i,j]
            precisions[t,i,j] += 1

            d_code = Int64(round((d_ground[t,i,j]-d_intermediate[t,i,j])*(2^precisions[t,i,j])/aeb))
            r_code = Int64(round((r_ground[t,i,j]-r_intermediate[t,i,j])*(2^precisions[t,i,j])/aeb))
            s_code = Int64(round((s_ground[t,i,j]-s_intermediate[t,i,j])*(2^precisions[t,i,j])/(sqrt(2)*aeb)))

            if d_code < -127 || d_code > 127
                d_code = 128
                d_final[t,i,j] = d_ground[t,i,j]
            else
                d_final[t,i,j] = d_intermediate[t,i,j] + aeb * d_code / (2^precisions[t,i,j])
            end
            d_codes[t,i,j] = d_code + 127

            if r_code < -127 || r_code > 127
                r_code = 128
                r_final[t,i,j] = r_ground[t,i,j]
            else
                r_final[t,i,j] = r_intermediate[t,i,j] + aeb * r_code / (2^precisions[t,i,j])
            end
            r_codes[t,i,j] = r_code + 127

            if s_code < -127 || s_code > 127
                s_code = 255
                s_final[t,i,j] = s_ground[t,i,j]
            else
                s_final[t,i,j] = s_intermediate[t,i,j] + sqrt(2) * aeb * s_code / (2^precisions[t,i,j])
            end
            s_codes[t,i,j] = s_code + 127

            if !angles_mandatory[t,i,j]
                θ_final[t,i,j] = θ_intermediate[t,i,j]
            end
        else
            if θ_quantized[t,i,j] == Inf
                θ_dif = θ_ground[t,i,j] - θ_intermediate[t,i,j]
                if θ_dif < 0
                    θ_dif += 2pi
                end

                θ_code = UInt8(round( θ_dif * (2^6-1) / 2pi ))
                if θ_code == 2^6-1
                    θ_code = 0
                end
                θ_codes[t,i,j] = θ_code
                θ_quantized[t,i,j] = θ_intermediate[t,i,j] + 2pi/(2^6-1)*θ_code
            end

            θ_final[t,i,j] = θ_quantized[t,i,j]
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

            # account for s possibly being negative
            if s_swap < -(sqrt(2)-1)*aeb
                s_swap += sqrt(2)*aeb
            elseif s_swap < 0
                s_swap += (sqrt(2)-1)*aeb
            end

            d_sign_swap = false
            r_sign_swap = false
            d_largest_swap = false
            r_over_s_swap = false
            degenerateCase = false

            # Check whether any modifications need to take place at all.
            modifications = false

            if precisions[coords...] != 0
                reconstructed = recomposeTensor(d_swap, r_swap, s_swap, θ_swap)
                d2, r2, s2, θ2 = decomposeTensor(reconstructed)
            else
                d2, r2, s2, θ2 = d_swap, r_swap, s_swap, θ_swap
            end

            eigenvectorRecon = classifyTensorEigenvector(r_swap, s_swap)
            eigenvectorGround = classifyTensorEigenvector(r_ground[coords...], s_ground[coords...])
            eigenvalueRecon = classifyTensorEigenvalue(d_swap, r_swap, s_swap)
            eigenvalueGround = classifyTensorEigenvalue(d_ground[coords...], r_ground[coords...], s_ground[coords...])

            if !( eigenvectorRecon == eigenvectorGround && eigenvalueRecon == eigenvalueGround )

                # check whether any of d, r, or s are equal such that swapping wouldn't work. If so, raise precision.
                degenerateCase = ((eigenvalueGround == POSITIVE_SCALING || eigenvalueGround == NEGATIVE_SCALING) && (abs(d_swap) == abs(r_swap) || abs(d_swap) == s_swap)) || abs(r_swap) == s_swap

                if !degenerateCase
                    modifications = true

                    # account for s possibly being out of range (max error is sqrt(2)aeb )
                    if s_swap - s_ground[coords...] > aeb
                        s_swap -= (sqrt(2)-1)*aeb
                        s_fix = 1
                    elseif s_swap - s_ground[coords...] < -aeb
                        s_swap += (sqrt(2)-1)*aeb
                        s_fix = 2
                    end

                    # eigenvector special cases and r_sign_swap

                    if eigenvectorGround == SYMMETRIC
                        eigenvector_special_cases[coords...] = 1
                    else
                        r_sign_swap = ( r_ground[coords...] > 0 ) ⊻ ( r_swap > 0 )                                    
                        if eigenvectorGround == PI_BY_4 || eigenvectorGround == MINUS_PI_BY_4
                            eigenvector_special_cases[coords...] = 2
                        end
                    end

                    if r_sign_swap
                        if r_swap < 0
                            r_swap += aeb
                        else
                            r_swap -= aeb
                        end
                    end

                    # d sign swap
                    d_sign_swap = ( abs(d_ground[coords...]) > abs(r_ground[coords...]) && abs(d_ground[coords...]) > s_ground[coords...] ) && ( (d_ground[coords...] > 0) ⊻ (d_swap > 0) )

                    if d_sign_swap
                        if d_swap < 0
                            d_swap += aeb
                        else
                            d_swap -= aeb
                        end
                    end

                    # magnitude swap codes

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
                        r_swap = 0
                    end
                end # end if not degenerate case
            end # end if the default classifications do not match

            # check whether case is degenerate, or whether no changes were made, or whether changes were made and those changes are valid.
            # then, tighten or terminate accordingly.
            if degenerateCase
                raise_precision(coords...)
            elseif !modifications && precisions[coords...] == 0 && θ_final[coords...] == θ_intermediate[coords...]
                pending = false
            else
                reconstructed = recomposeTensor(d_swap, r_swap, s_swap, θ_swap)
                d2, r2, s2, θ2 = decomposeTensor(reconstructed)
                if maximum(abs.(reconstructed - getTensor(tf, coords...))) <= aeb &&
                    classifyTensorEigenvector(r2, s2) == classifyTensorEigenvector(r_ground[coords...], s_ground[coords...]) &&
                    classifyTensorEigenvalue(d2, r2, s2) == classifyTensorEigenvalue(d_ground[coords...], r_ground[coords...], s_ground[coords...])

                    pending = false
                    sfix_codes[coords...] = s_fix
                    type_codes[coords...] = (d_sign_swap << 3) | (r_sign_swap << 2) | (d_largest_swap << 1) | (r_over_s_swap)

                    setTensor(tf2, coords..., reconstructed)
                else
                    raise_precision(coords...)
                end
            end

        end # end while pending

    end # end function process_point

    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]

                matrix = getTensor(tf2, t, i, j)
                d,r,s,θ = decomposeTensor(matrix)

                d_intermediate[t,i,j] = d
                d_final[t,i,j] = d
                r_intermediate[t,i,j] = r
                r_final[t,i,j] = r
                s_intermediate[t,i,j] = s
                s_final[t,i,j] = s
                θ_intermediate[t,i,j] = θ
                θ_final[t,i,j] = θ

            end
        end
    end


    stack = []

    visited = zeros(Int64, dims)

    for j in 1:dims[3]-1
        for i in 1:dims[2]-1
            for k in 0:1

                push!(stack, (i,j,Bool(k),true) )

                while length(stack) > 0
                    x,y,top,processNewVertices = pop!(stack)

                    # bottom left, bottom right, top left (corresponds to coords of bottom in order)
                    vertices_modified = [false,false,false]

                    crit_ground = getCircularPointType(tf, 1, x, y, top)
                    crit_intermediate = getCircularPointType(tf2, 1, x, y, top)

                    vertexCoords = getCellVertexCoords(1,x,y,top)

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
                            newVertices = []
                        end
                    end


                    # Single vertex: swap values into place.
                    for v in newVertices

                        processPoint(vertexCoords[v])

                    end # end for v in newVertices

                    # Edges



                end # end while length(stack) > 0

            end
        end
    end

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

    try
        run(`mkdir $output/test`)
    catch
    end

    saveArray("$output/test/row_1_col_1.dat", tf2.entries[1,1])
    saveArray("$output/test/row_1_col_2.dat", tf2.entries[1,2])
    saveArray("$output/test/row_2_col_1.dat", tf2.entries[2,1])
    saveArray("$output/test/row_2_col_2.dat", tf2.entries[2,2])

    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]
                base_codes[t,i,j] = type_codes[t,i,j] | (precisions[t,i,j] << 4)

                if precisions[t,i,j] >= 8
                    θ_and_sfix_codes[t,i,j] = 0

                    tensor = getTensor(tf, t, i, j)
                    push!(lossless_A, tensor[1,1])
                    push!(lossless_B, tensor[1,2])
                    push!(lossless_C, tensor[2,1])
                    push!(lossless_D, tensor[2,2])
                else
                    if θ_final[t,i,j] == θ_quantized[t,i,j]
                        θ_and_sfix_codes[t,i,j] = θ_codes[t,i,j] | (sfix_codes[t,i,j] << 6)
                    else
                        θ_and_sfix_codes[t,i,j] = (sfix_codes[t,i,j] << 6)
                    end

                    if d_codes[t,i,j] == 255
                        push!(lossless_d, d_ground[t,i,j])
                    end

                    if r_codes[t,i,j] == 255
                        push!(lossless_r, r_ground[t,i,j])
                    end

                    if s_codes[t,i,j] == 255
                        push!(lossless_s, s_ground[t,i,j])
                    end

                    if θ_codes[t,i,j] == 2^6-1
                        push!(lossless_θ, θ_ground[t,i,j] )
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
    removeIfExists("$output_file.tar.xz")

    run(`tar cvf $output_file.tar row_1_col_1.cmp row_1_col_2.cmp row_2_col_1.cmp row_2_col_2.cmp vals.bytes`)
    run(`xz -v9 $output_file.tar`)

    removeIfExists("$output_file.tar")

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

    return -1, -1

end

function compress2dSymmetric(containing_folder, dims, output_file, relative_error_bound, bits, output = "../output")
    tf, dtype = loadTensorField2dFromFolder(containing_folder, dims)

    # prepare derived attributes for compression

    max_entry = -Inf
    min_entry = Inf

    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]
                matrix = getTensor(tf, t, i, j)

                max_entry = max(max_entry, maximum(matrix))
                min_entry = min(min_entry, minimum(matrix))

            end
        end
    end

    aeb = relative_error_bound * (max_entry - min_entry)

    saveArray("$output/row_1_col_1_g.dat", Array{Float32}(tf.entries[1,1]))
    saveArray("$output/row_1_col_2_g.dat", Array{Float32}(tf.entries[1,2]))
    saveArray("$output/row_2_col_2_g.dat", Array{Float32}(tf.entries[2,2]))

    run(`../SZ3-master/build/bin/sz3 -f -i $output/row_1_col_1_g.dat -z $output/row_1_col_1.cmp -o $output/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/row_1_col_2_g.dat -z $output/row_1_col_2.cmp -o $output/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/row_2_col_2_g.dat -z $output/row_2_col_2.cmp -o $output/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`cp $output/row_1_col_2.dat $output/row_2_col_1.dat`)

    tf2, dtype2 = loadTensorField2dFromFolder(output, dims)
    stack::Array{Tuple{Int64,Int64,Bool}} = []

    codes = zeros(UInt64, dims)

    numProcess = 0

    for j in 1:dims[3]-1
        for i in 1:dims[2]-1
            for k in 0:1

                push!(stack, (i,j,Bool(k)))

                while length(stack) > 0
                    numProcess += 1
                    x,y,top = pop!(stack)

                    crit_ground = getCircularPointType(tf, 1, x, y, top)
                    crit_intermediate = getCircularPointType(tf2, 1, x, y, top)

                    if crit_ground != crit_intermediate

                        vertexCoords = getCellVertexCoords(1,x,y,top)

                        tensor1Ground = getTensor(tf, vertexCoords[1]...)
                        tensor2Ground = getTensor(tf, vertexCoords[2]...)
                        tensor3Ground = getTensor(tf, vertexCoords[3]...)

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

                        θg = [θ1g, θ2g, θ3g]
                        θr = [θ1r, θ2r, θ3r]
                        tg = [t1g, t2g, t3g]
                        tr = [t1r, t2r, t3r]
                        rr = [r1r, r2r, r3r]
                        tensorsGround = [tensor1Ground, tensor2Ground, tensor3Ground]

                        while crit_ground != crit_intermediate

                            if θe1 >= θe2 && θe1 >= θe3
                                idx = 1
                            elseif θe2 >= θe1 && θe2 >= θe3
                                idx = 2
                            else
                                idx = 3
                            end

                            lossless = (codes[vertexCoords[idx]...] != 0)

                            # that is, it hasn't been touched yet
                            if !lossless

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
                                    elseif idx == 2
                                        θe2 = abs(θnew - θg[idx])
                                    else
                                        θe3 = abs(θnew - θg[idx])
                                    end

                                    # no need to update the other values because the only other place this can go is lossless...

                                end

                            end

                            if lossless
                                codes[vertexCoords[idx]...] = 2^bits-1
                                newTensor = tensorsGround[idx] + (tr[idx]-tg[idx])*[1 0 ; 0 1] # only r and θ must be stored losslessly in this case.
                                setTensor(tf2, vertexCoords[idx]..., newTensor)

                                if idx == 1
                                    θe1 = 0
                                elseif idx == 2
                                    θe2 = 0
                                else
                                    θe3 = 0
                                end

                                # no need to update the other values because we're not going to touch this again.
                            end

                            crit_intermediate = getCircularPointType(tf2, 1, x, y, top)

                        end

                        # requeue up any cells that must be hit after edits

                        if top
                            push!(stack, (x,y,false))
                            if x != 1
                                push!(stack, (x-1,y,true))
                            end
                            if y != 1
                                if x != dims[2]-1
                                    push!(stack, (x+1,y-1,true))
                                    push!(stack, (x+1,y-1,false))
                                end

                                push!(stack, (x,y-1,true))

                            end
                        else

                            if x != 1
                                push!(stack, (x-1,y,true))
                                push!(stack, (x-1,y,false))
                            end

                            if y != 1
                                if x != dims[2]-1
                                    push!(stack, (x+1,y-1,true))
                                    push!(stack, (x+1,y-1,false))
                                end

                                push!(stack, (x,y-1,true))
                                push!(stack, (x,y-1,false))

                                if x != 1
                                    push!(stack, (x-1,y-1,true))
                                end
                            end

                        end

                        push!(stack, (x,y,top))

                    end # end if crit_ground != crit_intermediate

                end # end while length(stack) > 0

            end # end for k in 0:1
        end # end for i
    end # end for j

    losslessValues::Vector{Float64} = []
    for j in 1:dims[3]
        for i in 1:dims[2]
            if codes[1,i,j] == 2^bits-1
                next_lossless = getTensor(tf, 1, i, j)
                push!(losslessValues, (next_lossless[1,1]-next_lossless[2,2])/2 )
                push!(losslessValues, next_lossless[1,2])
            end
        end
    end
    codeBytes = huffmanEncode(vec(codes))

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])     
    write(vals_file, aeb)
    write(vals_file, length(codeBytes))
    write(vals_file, codeBytes)
    write(vals_file, losslessValues)
    close(vals_file)

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.xz")

    run(`tar cvf $output_file.tar row_1_col_1.cmp row_1_col_2.cmp row_2_col_2.cmp vals.bytes`)
    run(`xz -v9 $output_file.tar`)

    removeIfExists("$output_file.tar")

    cd(cwd)

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/row_1_col_1.dat")
    remove("$output/row_1_col_2.dat")
    remove("$output/row_2_col_1.dat")
    remove("$output/row_2_col_2.dat")
    remove("$output/vals.bytes")
end

end