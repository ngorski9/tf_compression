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

function compress2dNaive(containing_folder, dims, output_file, relative_error_bound, output = "../output", verbose=false)
    tf, dtype = loadTensorField2dFromFolder(containing_folder, dims)

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

function compress2d(containing_folder, dims, output_file, relative_error_bound, edgeEB=1.0, output="../output", verbose=false)
    tf, dtype = loadTensorField2dFromFolder(containing_folder, dims)
    
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
                if (i,j,t) == (47,30,1)
                    println("set 1")
                end

            else
                precisions[i,j,t] += 1
                d_code = Int64(round((d_ground[i,j,t]-d_intermediate[i,j,t])*(2^precisions[i,j,t])/aeb))
                r_code = Int64(round((r_ground[i,j,t]-r_intermediate[i,j,t])*(2^precisions[i,j,t])/aeb))
                s_code = Int64(round((s_ground[i,j,t]-s_intermediate[i,j,t])*(2^precisions[i,j,t])/(sqrt(2)*aeb)))

                if d_code < -127 || d_code > 127
                    d_code = 255
                    d_final[i,j,t] = d_ground[i,j,t]
                else
                    d_final[i,j,t] = d_intermediate[i,j,t] + aeb * d_code / (2^precisions[i,j,t])
                end
                d_codes[i,j,t] = d_code + 127

                if r_code < -127 || r_code > 127
                    r_code = 255
                    r_final[i,j,t] = r_ground[i,j,t]
                else
                    r_final[i,j,t] = r_intermediate[i,j,t] + aeb * r_code / (2^precisions[i,j,t])
                end
                r_codes[i,j,t] = r_code + 127

                if s_code < -127 || s_code > 127
                    s_code = 255
                    s_final[i,j,t] = s_ground[i,j,t]
                else
                    s_final[i,j,t] = s_intermediate[i,j,t] + sqrt(2) * aeb * s_code / (2^precisions[i,j,t])
                end
                s_codes[i,j,t] = s_code + 127

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

            eigenvectorRecon = classifyTensorEigenvector(r_swap, s_swap)
            eigenvectorGround = classifyTensorEigenvector(r_ground[coords...], s_ground[coords...])
            eigenvalueRecon = classifyTensorEigenvalue(d_swap, r_swap, s_swap)
            eigenvalueGround = classifyTensorEigenvalue(d_ground[coords...], r_ground[coords...], s_ground[coords...])

            if !( eigenvectorRecon == eigenvectorGround && eigenvalueRecon == eigenvalueGround && maximum(abs.(getTensor(tf,coords...)-getTensor(tf2,coords...))) <= aeb)

                # check whether any of d, r, or s are equal such that swapping wouldn't work. If so, raise precision.
                degenerateCase = ((eigenvalueGround == POSITIVE_SCALING || eigenvalueGround == NEGATIVE_SCALING) && (abs(d_swap) == abs(r_swap) || abs(d_swap) == s_swap)) || abs(r_swap) == s_swap

                if !degenerateCase
                    modifications = true

                    if precisions[coords...] == 0
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

                    if eigenvectorGround == SYMMETRIC && eigenvectorRecon != eigenvectorGround
                        eigenvector_special_cases[coords...] = 1
                    else
                        r_sign_swap = ( r_ground[coords...] > 0 ) ⊻ ( r_swap > 0 )
                        if (eigenvectorGround == PI_BY_4 || eigenvectorGround == MINUS_PI_BY_4) && eigenvectorRecon != eigenvectorGround
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
                        r_swap = 0.0
                    end
                end # end if not degenerate case
            end # end if the default classifications do not match

            # check whether case is degenerate, or whether no changes were made, or whether changes were made and those changes are valid.
            # then, tighten or terminate accordingly.
            if degenerateCase
                raise_precision(coords...)
            elseif precisions[coords...] >= 8 || (!modifications && precisions[coords...] == 0 && θ_final[coords...] == θ_intermediate[coords...])
                pending = false
                # if coords == (249,32,1)
                #     println("skip")
                # end                
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
                    if coords == (47,30,1)
                        println("set 2")
                    end
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

    stack = []

    for t in 1:dims[3]   
        for j in 1:dims[2]-1
            for i in 1:dims[1]-1
                for k in 0:1

                    push!(stack, (i,j,Bool(k),true) )

                    while length(stack) > 0
                        x,y,top,processNewVertices = pop!(stack)

                        # bottom left, bottom right, top left (corresponds to coords of bottom in order) then top right
                        vertices_modified = [false,false,false,false]

                        vertexCoords = getCellVertexCoords(x,y,t,top)

                        if processNewVertices
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

                                if vertexCoords[v] == (47,30,1)
                                    println("initial process")
                                end
                            end

                        end

                        # Edges
                        if top
                            newEdges = [(1,3), (2,3)]
                        else
                            if x == 1
                                if y == 1
                                    newEdges = [(1,2), (1,3), (2,3)]
                                else
                                    newEdges = [(1,3), (2,3)]
                                end
                            elseif y == 1
                                newEdges = [(1,2), (2,3)]
                            else
                                newEdges = [(2,3)]
                            end
                        end

                        # cells


                        for e in newEdges
                            # if x == 46 && y == 30 && !top
                            #     println("===============")
                            #     println("checking an edge")
                            #     println((vertexCoords[e[1]],vertexCoords[e[2]]))

                            #     println(edgesMatch( getTensor(tf, vertexCoords[e[1]]...), getTensor(tf, vertexCoords[e[2]]...), getTensor(tf2, vertexCoords[e[1]]...), getTensor(tf2, vertexCoords[e[2]]...), edgeEB ))
                            #     println(edgesMatch( getTensor(tf, vertexCoords[e[2]]...), getTensor(tf, vertexCoords[e[1]]...), getTensor(tf2, vertexCoords[e[2]]...), getTensor(tf2, vertexCoords[e[1]]...), edgeEB ))

                            #     println(getTensor(tf2, vertexCoords[e[1]]...))
                            #     println(getTensor(tf2, vertexCoords[e[2]]...))
                            #     a1,b1,c1,d1 = classifyEdge(getTensor(tf2, vertexCoords[e[1]]...), getTensor(tf2, vertexCoords[e[2]]...), true)
                            #     a2,b2,c2,d2 = classifyEdge(getTensor(tf2, vertexCoords[e[2]]...), getTensor(tf2, vertexCoords[e[1]]...), true)
                            #     println((a1,b1,c1,d1))
                            #     println((a2,b2,c2,d2))
                            #     println("------------")
                            #     println(getTensor(tf, vertexCoords[e[1]]...))
                            #     println(getTensor(tf, vertexCoords[e[2]]...))                                
                            #     a1,b1,c1,d1 = classifyEdge(getTensor(tf, vertexCoords[e[1]]...), getTensor(tf, vertexCoords[e[2]]...), true)
                            #     a2,b2,c2,d2 = classifyEdge(getTensor(tf, vertexCoords[e[2]]...), getTensor(tf, vertexCoords[e[1]]...), true)                                
                            #     println((a1,b1,c1,d1))
                            #     println((a2,b2,c2,d2))                                
                            # end

                            while !edgesMatch( getTensor(tf, vertexCoords[e[1]]...), getTensor(tf, vertexCoords[e[2]]...), getTensor(tf2, vertexCoords[e[1]]...), getTensor(tf2, vertexCoords[e[2]]...), edgeEB )
                                raise_precision(vertexCoords[e[1]]...)
                                processPoint(vertexCoords[e[1]])

                                raise_precision(vertexCoords[e[2]]...)
                                processPoint(vertexCoords[e[2]])

                                if vertexCoords[e[1]] == (47,30,1) || vertexCoords[e[2]] == (47,30,1)
                                    println("process edge")
                                end

                                # record which vertices was modified in order to check previous cells.
                                if top
                                    if e[1] == 1 || e[2] == 1
                                        vertices_modified[3] = true
                                    end

                                    if e[1] == 2 || e[2] == 2
                                        vertices_modified[2] = true
                                    end

                                    if e[1] == 3 || e[2] == 3
                                        vertices_modified[4] = true
                                    end
                                else
                                    vertices_modified[e[1]] = true
                                    vertices_modified[e[2]] = true
                                end
                            end

                        end

                        # Process individual cells to check circular points.
                        if getCircularPointType(tf, x, y, t, top) != getCircularPointType(tf2, x, y, t, top)

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

                            while getCircularPointType(tf, x, y, t, top) != getCircularPointType(tf2, x, y, t, top)
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

                                    if vertexCoords[1] == (47,30,1) || vertexCoords[2] == (47,30,1) || vertexCoords[3] == (47,30,1)
                                        println("set 3")
                                    end

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

                                        if vertexCoords[changeTensor] == (47,30,1)
                                            println("set 4")
                                        end
                                                                                                    
                                        processPoint(vertexCoords[changeTensor])
                                        if vertexCoords[changeTensor] == (47,30,1)
                                            println("process angle")
                                        end
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

                        if vertices_modified[1] || vertices_modified[2] || vertices_modified[3] || vertices_modified[4]
                            push!(stack, (x,y,top,false))
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

                    end # end while length(stack) > 0

                end
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

    try
        run(`mkdir $output/test`)
    catch
    end
    saveArray("$output/test/row_1_col_1.dat", tf2.entries[1,1])
    saveArray("$output/test/row_1_col_2.dat", tf2.entries[1,2])
    saveArray("$output/test/row_2_col_1.dat", tf2.entries[2,1])
    saveArray("$output/test/row_2_col_2.dat", tf2.entries[2,2])

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
    full_lossless = zeros(UInt64, dims)
    processed = zeros(Bool, dims)

    numProcess = 0

    checkLoc = (24,12,11,false)
    checking = false

    for t in 1:dims[3]
        for j in 1:dims[2]-1
            for i in 1:dims[1]-1
                for k in 0:1

                    push!(stack, (i,j,Bool(k)))

                    while length(stack) > 0
                        numProcess += 1
                        x,y,top = pop!(stack)

                        if checking
                            if (x,y,t,top) == checkLoc
                                println("self")
                            elseif t == checkLoc[3] && abs(x-checkLoc[1]) <= 1 && abs(y-checkLoc[2]) <= 1
                                println(("neighbor", (x,y,t,top)))
                            end
                        end

                        crit_ground = getCircularPointType(tf, x, y, t, top)
                        crit_intermediate = getCircularPointType(tf2, x, y, t, top)

                        if crit_ground != crit_intermediate

                            if checking && t == checkLoc[3] && abs(x-checkLoc[1]) <= 1 && abs(y-checkLoc[2]) <= 1
                                println("edit")
                            end

                            vertexCoords = getCellVertexCoords(x,y,t,top)

                            tensor1Ground = getTensor(tf, vertexCoords[1]...)
                            tensor2Ground = getTensor(tf, vertexCoords[2]...)
                            tensor3Ground = getTensor(tf, vertexCoords[3]...)

                            # do full lossless checks
                            if tensor1Ground == [0.0 0.0 ; 0.0 0.0]
                                full_lossless[vertexCoords[1]...] = 1
                                setTensor(tf2, vertexCoords[1]..., tensor1Ground)
                            end

                            if tensor2Ground == [0.0 0.0 ; 0.0 0.0]
                                full_lossless[vertexCoords[2]...] = 1
                                setTensor(tf2, vertexCoords[2]..., tensor2Ground)
                            end

                            if tensor3Ground == [0.0 0.0 ; 0.0 0.0]
                                full_lossless[vertexCoords[3]...] = 1
                                setTensor(tf2, vertexCoords[3]..., tensor3Ground)
                            end

                            # check if there is still an issue after going full lossless
                            crit_intermediate = getCircularPointType(tf2, x, y, t, top)

                            if crit_ground != crit_intermediate

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

                                        else
                                            ll[idx] = true
                                            codes[vertexCoords[idx]...] = 2^bits-1
                                            newTensor = tensorsGround[idx] + (tr[idx]-tg[idx])*[1 0 ; 0 1] # only r and θ must be stored losslessly in this case.
                                            setTensor(tf2, vertexCoords[idx]..., newTensor)

                                            if idx == 1
                                                θe1 = 0.0
                                            elseif idx == 2
                                                θe2 = 0.0
                                            else
                                                θe3 = 0.0
                                            end

                                            # no need to update the other values because we're not going to touch this again.
                                        end
                                    end

                                    crit_intermediate = getCircularPointType(tf2, x, y, t, top)

                                end

                                # requeue up any cells that must be hit after edits

                                if top

                                    # future cells (if we are modifying a past cell)
                                    if x != dims[1]-1 && ((y+1 < j) || (y+1 == j && x+1 <= i))
                                        push!(stack, (x+1,y+1,false))
                                    end

                                    if y+1 < j || (y+1 == j && x <= i)
                                        push!(stack, (x,y+1,true))
                                        push!(stack, (x,y+1,false))
                                    end

                                    if x != 1 && ((y+1 < j) || (y+1 == j && x-1 <= i))
                                        push!(stack, (x-1,y+1,true))                                    
                                        push!(stack, (x-1,y+1,false))
                                    end

                                    if x != dims[1]-1 && ((y < j) || (y == j && x+1 <= i))
                                        push!(stack, (x+1,y,true))
                                        push!(stack, (x+1,y,false))
                                    end

                                    # past cells
                                    push!(stack, (x,y,false))
                                    if x != 1
                                        push!(stack, (x-1,y,true))
                                    end
                                    if y != 1
                                        if x != dims[1]-1
                                            push!(stack, (x+1,y-1,true))
                                            push!(stack, (x+1,y-1,false))
                                        end

                                        push!(stack, (x,y-1,true))

                                    end
                                else

                                    # future cells (if we are modifying a past cell)
                                    if (y+1 < j) || (y+1 == j && x <= i)
                                        push!(stack, (x,y+1,false))
                                    end

                                    if x != 1 && ((y+1 < j) || (y+1 == j && x-1 <= i))
                                        push!(stack, (x-1,y+1,true))
                                        push!(stack, (x-1,y+1,false))
                                    end

                                    if x != dims[1] - 1 && ((y < j) || (y == j && x+1 <= i))
                                        push!(stack, (x+1,y,false))
                                    end

                                    if y < j || (y == j && x < i) || (y == j && x == i && k == 1)
                                        push!(stack, (x,y,true))
                                    end

                                    # past cells
                                    if x != 1
                                        push!(stack, (x-1,y,true))
                                        push!(stack, (x-1,y,false))
                                    end

                                    if y != 1
                                        if x != dims[1]-1
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

                            end # end other if crit_ground != crit intermediate (after lossless check)

                        end # end if crit_ground != crit_intermediate

                    end # end while length(stack) > 0

                end # end for k in 0:1
            end # end for i
        end # end for j
    end # end for t

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
                    push!(losslessValues, next_lossless[1,1])
                    push!(losslessValues, next_lossless[1,2])
                    push!(losslessValues, next_lossless[2,2])
                elseif codes[i,j,t] == 2^bits-1
                    next_lossless = getTensor(tf, i, j, t)
                    push!(losslessValues, (next_lossless[1,1]-next_lossless[2,2])/2 )
                    push!(losslessValues, next_lossless[1,2])
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