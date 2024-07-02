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

mutable struct ProcessDataSymmetric
    tf_ground::SymmetricTensorField2d
    tf_reconstructed::SymmetricTensorField2d
    θ_ground::FloatArray
    r_intermediate::Array{Float32}
    trace_intermediate::Array{Float32}
    r_bound::Float64
    codes::Array{Int64}
end

struct ProcessData
    aeb::Float64 # absolute error bound
    ceb::Float64 # compressor error bound    
    tf::TensorField2d
    tf_reconstructed::TensorField2d
    d_ground::Array{Float32} # These are always 32 bit conversions. For a 64 bit version, decompose a tensor in "tf"
    r_ground::Array{Float32}
    s_ground::Array{Float32}
    θ_ground::Array{Float32}
    d_intermediate::FloatArray
    r_intermediate::FloatArray
    s_intermediate::FloatArray
    θ_intermediate::FloatArray
    d_precision::Array{Int64}
    r_precision::Array{Int64}
    s_precision::Array{Int64}
    d_quantization::Array{Int64}
    r_quantization::Array{Int64}
    s_quantization::Array{Int64}
    codes::Array{Int64}
end

function compress2dNaive(containing_folder, dims, output_file, relative_error_bound, output = "../output")
    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_1.dat -z $output/row_1_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M REL $relative_error_bound`)
    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_1_col_2.dat -z $output/row_1_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M REL $relative_error_bound`)
    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_2_col_1.dat -z $output/row_2_col_1.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M REL $relative_error_bound`)    
    run(`../SZ3-master/build/bin/sz3 -d -i $containing_folder/row_2_col_2.dat -z $output/row_2_col_2.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M REL $relative_error_bound`)

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

# returns true if the point was processed (e.g. its code was 0 and then it was updated)
# and false otherwise.
function processPoint( pd::ProcessData, t::Int64, x::Int64, y::Int64 )

    if pd.codes[t,x,y] == 0
        processPointDRS( pd, t, x, y )
        return true
    else
        return false
    end

end

# process the d, r, and s values of a point
function processPointDRS( pd::ProcessData, t::Int64, x::Int64, y::Int64 )

    code = 1
    groundTensor = getTensor(pd.tf, t, x, y)

    d_ = pd.d_intermediate[t, x, y]
    r_ = pd.r_intermediate[t, x, y]
    s_ = pd.s_intermediate[t, x, y]
    θ_ = pd.θ_intermediate[t, x, y]

    # Use these manipulations to avoid numerical precision issues
    d_ground64, r_ground64, s_ground64, θ_ground64 = decomposeTensor(groundTensor)

    reconstructedMatrix = recomposeTensor(d_, r_, s_, θ_)
    d_recompose, r_recompose, s_recompose, θ_recompose = decomposeTensor(reconstructedMatrix)

    eVectorGround = classifyTensorEigenvector(r_ground64, s_ground64)
    eVectorIntermediate = classifyTensorEigenvector(r_recompose, s_recompose)
    
    eValueGround = classifyTensorEigenvalue(d_ground64, r_ground64, s_ground64)
    eValueIntermediate = classifyTensorEigenvalue(d_recompose, r_recompose, s_recompose)

    # update counter
    # if eVectorGround != eVectorIntermediate || eValueGround != eValueIntermediate

    # Modify vertex if either classification doesn't match
    if eVectorGround != eVectorIntermediate || eValueGround != eValueIntermediate

        if eVectorGround != eVectorIntermediate
            code *= CODE_CHANGE_EIGENVECTOR^eVectorGround
        end

        if eValueGround != eValueIntermediate
            code *= CODE_CHANGE_EIGENVALUE^eValueGround
        end

        d_ground32 = pd.d_ground[t, vertex_i, vertex_j]
        r_ground32 = pd.r_ground[t, vertex_i, vertex_j]
        s_ground32 = pd.s_ground[t, vertex_i, vertex_j]

        # Check if we have any values equal to each other, which causes the swapping algorithm to wig out on us.
        if abs( abs(r_) - abs(s_) ) < ϵ && abs( abs(r_ground64) - abs(s_ground64) ) > ϵ
            if abs(s_ground32) != abs(r_)
                code *= CODE_LOSSLESS_S
                s_ = s_ground32
            else
                code *= CODE_LOSSLESS_R
                r_ = r_ground32
            end
        end

        if eValueGround != CLOCKWISE_ROTATION && eValueGround != COUNTERCLOCKWISE_ROTATION &&
            abs( abs(d_) - abs(s_) ) < ϵ && abs( abs(d_ground64) - abs(s_ground64) ) > ϵ
            if abs(s_ground32) != abs(d_) && code % CODE_LOSSLESS_S != 0
                code *= CODE_LOSSLESS_S
                s_ = s_ground32
            else
                code *= CODE_LOSSLESS_D
                d_ = d_ground32
            end
        end

        if eValueGround != ANISOTROPIC_STRETCHING &&
            abs( abs(d_) - abs(r_) ) < ϵ && abs( abs(d_ground64) - abs(r_ground64) ) > ϵ
            if abs(r_ground32) != abs(d_) && code % CODE_LOSSLESS_R != 0
                code *= CODE_LOSSLESS_R
                r_ = r_ground32
            elseif code % CODE_LOSSLESS_D != 0
                code *= CODE_LOSSLESS_D
                d_ = d_ground32
            end
        end

        # Further adjust entries based on any lossless settings
        d_, r_, s_, θ_ = adjustDecompositionEntries(d_, r_, s_, θ_, pd.ceb, code)
        reconstructedMatrix = recomposeTensor(d_, r_, s_, θ_)
        d_, r_, s_, θ_ = decomposeTensor(reconstructedMatrix)

        worked = true

        # Check if these adjustments worked. If not, store all entries losslessly.
        if classifyTensorEigenvector(r_, s_) != eVectorGround || classifyTensorEigenvalue(d_, r_, s_) != eValueGround
            worked = false
        end

        return (worked, code, d_, r_, s_)

    end

    codes[t,vertex_i,vertex_j] = code
    setTensor(tf_reconstructed, t, vertex_i, vertex_j, reconstructedMatrix)

    return true

end

function compress2d(containing_folder, dims, output_file, relative_error_bound, output="../output")

    # Fill out base arrays & calculate error bounds.

    tf, dtype = loadTensorField2dFromFolder(containing_folder, dims)

    A_ground = zeros(Float32, dims)
    B_ground = zeros(Float32, dims)
    C_ground = zeros(Float32, dims)
    D_ground = zeros(Float32, dims)


    d_ground = zeros(Float32, dims)
    r_ground = zeros(Float32, dims)
    s_ground = zeros(Float32, dims)
    θ_ground = zeros(Float32, dims)
    
    max_entry = -Inf
    min_entry = Inf

    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]
                tensor = getTensor(tf, t, i, j)

                A_ground[t,i,j] = tensor[1,1]
                B_ground[t,i,j] = tensor[1,2]
                C_ground[t,i,j] = tensor[2,1]
                D_ground[t,i,j] = tensor[2,2]

                max_entry = max(max_entry, maximum(tensor))
                min_entry = min(min_entry, minimum(tensor))

                d, r, s, θ = decomposeTensor(tensor)

                d_ground[t,i,j] = d
                r_ground[t,i,j] = r
                s_ground[t,i,j] = s
                θ_ground[t,i,j] = θ

            end
        end
    end

    #aeb = absolute error bound
    aeb = relative_error_bound * (max_entry - min_entry)
    #ceb = compressor error bound
    ceb = aeb * (2/3)

    saveArray("$output/a.dat", A_ground)
    saveArray("$output/b.dat", B_ground)
    saveArray("$output/c.dat", C_ground)
    saveArray("$output/d.dat", D_ground)

    # Initial compression

    run(`../SZ3-master/build/bin/sz3 -f -i $output/a.dat -z $output/a.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $ceb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/b.dat -z $output/b.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $ceb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/c.dat -z $output/c.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $ceb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/d.dat -z $output/d.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $ceb`)

    # Initial decompression

    run(`../SZ3-master/build/bin/sz3 -f -z $output/a.cmp -o $output/a_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $ceb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/b.cmp -o $output/b_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $ceb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/c.cmp -o $output/c_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $ceb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/d.cmp -o $output/d_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $ceb`)

    A_intermediate = loadArray("$output/a_intermediate.dat", Float32, dims)
    B_intermediate = loadArray("$output/b_intermediate.dat", Float32, dims)
    C_intermediate = loadArray("$output/c_intermediate.dat", Float32, dims)
    D_intermediate = loadArray("$output/d_intermediate.dat", Float32, dims)

    d_intermediate = zeros(Float32, dims)
    r_intermediate = zeros(Float32, dims)
    s_intermediate = zeros(Float32, dims)
    θ_intermediate = zeros(Float32, dims)

    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]
                tensor = [A_intermediate[t,i,j] B_intermediate[t,i,j] ; C_intermediate[t,i,j] D_intermediate[t,i,j]]
                d,r,s,θ = decomposeTensor(tensor)
                
                d_intermediate[t,i,j] = d
                r_intermediate[t,i,j] = r
                s_intermediate[t,i,j] = s
                θ_intermediate[t,i,j] = θ

            end
        end
    end

    tf_reconstructed = zeroTensorField(dtype, dims)

    # Error processing

    T = dims[1]
    x = dims[2]-1
    y = dims[3]-1
    cellsToVisit::Array{Tuple{Int64, Int64, Bool}} = [] 
    codes = zeros(dims)

    for j_ in 1:y
        for i_ in 1:x
            for t in 1:T
                for k in 0:1
                    push!(cellsToVisit, (i_,j_,Bool(k)))

                    while length(cellsToVisit) != 0
                        cell_i, cell_j, cell_top = pop!(cellsToVisit)

                        # Figure out which new vertices need to be processed for the given cell
                        vertexCoords = getCellVertexCoords(t, cell_i, cell_j, cell_top)

                        # Store which vertices we have altered on this pass.
                        modified_vertices = [false, false, false]

                        if cell_top
                            newVertices = [3]
                        elseif cell_i == 1
                            if cell_j == 1
                                newVertices = [1,2,3]
                            else
                                newVertices = [2]
                            end
                        elseif cell_j == 1
                            newVertices = [2]
                        else
                            newVertices = []
                        end

                        # Process vertices
                        for newVertex in newVertices

                            _, vertex_i, vertex_j = vertexCoords[newVertex]

                            # Only proceed if the vertex hasn't been processed yet. This is because
                            # if a vertex has already been processed, its eigenvalue and eigenvector
                            # classes are not affected. The only alternative is that it ends up getting stored losslessly.
                            # May change this later if edge classes make meaningful changes to these.
                            if codes[t, vertex_i, vertex_j] == 0

                                modified_vertices[newVertex] = true

                                code = 1
                                groundTensor = getTensor(tf, t, vertex_i, vertex_j)

                                d_ = d_intermediate[t, vertex_i, vertex_j]
                                r_ = r_intermediate[t, vertex_i, vertex_j]
                                s_ = s_intermediate[t, vertex_i, vertex_j]
                                θ_ = θ_intermediate[t, vertex_i, vertex_j]

                                # Use these manipulations to avoid numerical precision issues
                                d_ground64, r_ground64, s_ground64, θ_ground64 = decomposeTensor(groundTensor)

                                reconstructedMatrix = recomposeTensor(d_, r_, s_, θ_)
                                d_recompose, r_recompose, s_recompose, θ_recompose = decomposeTensor(reconstructedMatrix)

                                eVectorGround = classifyTensorEigenvector(r_ground64, s_ground64)
                                eVectorIntermediate = classifyTensorEigenvector(r_recompose, s_recompose)
                                
                                eValueGround = classifyTensorEigenvalue(d_ground64, r_ground64, s_ground64)
                                eValueIntermediate = classifyTensorEigenvalue(d_recompose, r_recompose, s_recompose)

                                # update counter
                                # if eVectorGround != eVectorIntermediate || eValueGround != eValueIntermediate

                                # Modify vertex if either classification doesn't match
                                if eVectorGround != eVectorIntermediate || eValueGround != eValueIntermediate

                                    if eVectorGround != eVectorIntermediate
                                        code *= CODE_CHANGE_EIGENVECTOR^eVectorGround
                                    end

                                    if eValueGround != eValueIntermediate
                                        code *= CODE_CHANGE_EIGENVALUE^eValueGround
                                    end

                                    d_ground32 = d_ground[t, vertex_i, vertex_j]
                                    r_ground32 = r_ground[t, vertex_i, vertex_j]
                                    s_ground32 = s_ground[t, vertex_i, vertex_j]

                                    # Check if we have any values equal to each other, which causes the swapping algorithm to wig out on us.
                                    if abs( abs(r_) - abs(s_) ) < ϵ && abs( abs(r_ground64) - abs(s_ground64) ) > ϵ
                                        if abs(s_ground32) != abs(r_)
                                            code *= CODE_LOSSLESS_S
                                            s_ = s_ground32
                                        else
                                            code *= CODE_LOSSLESS_R
                                            r_ = r_ground32
                                        end
                                    end

                                    if eValueGround != CLOCKWISE_ROTATION && eValueGround != COUNTERCLOCKWISE_ROTATION &&
                                        abs( abs(d_) - abs(s_) ) < ϵ && abs( abs(d_ground64) - abs(s_ground64) ) > ϵ
                                        if abs(s_ground32) != abs(d_) && code % CODE_LOSSLESS_S != 0
                                            code *= CODE_LOSSLESS_S
                                            s_ = s_ground32
                                        else
                                            code *= CODE_LOSSLESS_D
                                            d_ = d_ground32
                                        end
                                    end

                                    if eValueGround != ANISOTROPIC_STRETCHING &&
                                        abs( abs(d_) - abs(r_) ) < ϵ && abs( abs(d_ground64) - abs(r_ground64) ) > ϵ
                                        if abs(r_ground32) != abs(d_) && code % CODE_LOSSLESS_R != 0
                                            code *= CODE_LOSSLESS_R
                                            r_ = r_ground32
                                        elseif code % CODE_LOSSLESS_D != 0
                                            code *= CODE_LOSSLESS_D
                                            d_ = d_ground32
                                        end
                                    end

                                    # Further adjust entries based on any lossless settings
                                    d_, r_, s_, θ_ = adjustDecompositionEntries(d_, r_, s_, θ_, ceb, code)
                                    reconstructedMatrix = recomposeTensor(d_, r_, s_, θ_)
                                    d_, r_, s_, θ_ = decomposeTensor(reconstructedMatrix)

                                    # Check if these adjustments worked. If not, store all entries losslessly.
                                    if classifyTensorEigenvector(r_, s_) != eVectorGround || classifyTensorEigenvalue(d_, r_, s_) != eValueGround
                                        code = CODE_LOSSLESS_FULL
                                        reconstructedMatrix = Matrix{Float32}(groundTensor)
                                        d_ = d_ground32
                                        r_ = r_ground32
                                        s_ = s_ground32
                                        θ_ = θ_ground[t, vertex_i, vertex_j]
                                    end

                                    if classifyTensorEigenvector(r_, s_) != eVectorGround || classifyTensorEigenvalue(d_, r_, s_) != eValueGround
                                        println("problem!")
                                        exit()
                                    end

                                    if code != 1
                                        d_intermediate[t, vertex_i, vertex_j] = d_
                                        r_intermediate[t, vertex_i, vertex_j] = r_
                                        s_intermediate[t, vertex_i, vertex_j] = s_
                                        θ_intermediate[t, vertex_i, vertex_j] = θ_
                                    end

                                end

                                codes[t,vertex_i,vertex_j] = code
                                setTensor(tf_reconstructed, t, vertex_i, vertex_j, reconstructedMatrix)
                            end

                        end 

                        # Figure out which new edges correspond to the given cell
                        if cell_top
                            newEdges = [(3,1), (2,3)]
                        elseif cell_i == 1
                            if cell_j == 1
                                newEdges = [(1,2), (2,3), (3,1)]
                            else
                                newEdges = [(2,3), (3,1)]
                            end
                        elseif cell_j == 1
                            newEdges = [(1,2), (2,3)]
                        else
                            newEdges = [(2,3)]
                        end

                        keepChecking = true

                        # edges
                        num = 0
                        while keepChecking
                            keepChecking = false
                            num += 1
                        for edge in newEdges

                            _, i1,j1 = vertexCoords[edge[1]]
                            _, i2,j2 = vertexCoords[edge[2]]

                            t11 = getTensor(tf, t, i1, j1)
                            t12 = getTensor(tf_reconstructed, t, i1, j1)
                            t21 = getTensor(tf, t, i2, j2)
                            t22 = getTensor(tf_reconstructed, t, i2, j2)

                            class1 = classifyEdgeEigenvalue(t11, t21)
                            class2 = classifyEdgeEigenvalue(t12, t22)

                            if class1 != class2
                                keepChecking = true
                                modified_vertices[edge[1]] = true
                                modified_vertices[edge[2]] = true
                                code = codes[t,i1,j1]
                                if code % CODE_LOSSLESS_ANGLE != 0 && code != CODE_LOSSLESS_FULL && code != CODE_LOSSLESS_FULL_64
                                    codes[t, i1, j1] *= CODE_LOSSLESS_ANGLE
                                    θ_intermediate[t,i1,j1] = θ_ground[t,i1,j1]
                                    setTensor( tf_reconstructed, t, i1, j1, 
                                        recomposeTensor( d_intermediate[t,i1,j1],
                                                         r_intermediate[t,i1,j1],
                                                         s_intermediate[t,i1,j1],
                                                         θ_intermediate[t,i1,j1] )
                                    )
                                end

                                code = codes[t,i2,j2]
                                if code % CODE_LOSSLESS_ANGLE != 0 && code != CODE_LOSSLESS_FULL && code != CODE_LOSSLESS_FULL_64
                                    codes[t, i2, j2] *= CODE_LOSSLESS_ANGLE
                                    θ_intermediate[t,i2,j2] = θ_ground[t,i2,j2]
                                    setTensor( tf_reconstructed, t, i2, j2, 
                                        recomposeTensor( d_intermediate[t,i2,j2],
                                                         r_intermediate[t,i2,j2],
                                                         s_intermediate[t,i2,j2],
                                                         θ_intermediate[t,i2,j2] )
                                    )
                                end

                                class2 = classifyEdgeEigenvalue(getTensor(tf_reconstructed, t, i1, j1), getTensor(tf_reconstructed, t, i2, j2))
                                if class1 != class2
                                    codes[t, i1, j1] = CODE_LOSSLESS_FULL_64
                                    setTensor( tf_reconstructed, t, i1, j1, getTensor(tf, t, i1, j1) )

                                    codes[t, i2, j2] = CODE_LOSSLESS_FULL_64
                                    setTensor( tf_reconstructed, t, i2, j2, getTensor(tf, t, i2, j2) )
                                end

                            end

                            t11 = getTensor(tf, t, i1, j1)
                            t12 = getTensor(tf_reconstructed, t, i1, j1)
                            t21 = getTensor(tf, t, i2, j2)
                            t22 = getTensor(tf_reconstructed, t, i2, j2)

                            class1 = classifyEdgeEigenvalue(t11, t21)
                            class2 = classifyEdgeEigenvalue(t12, t22)

                            if class1 != class2
                                keepChecking = true
                            end

                        end
                        end

                        # cell
                        t1Ground = getTensor(tf, vertexCoords[1]...)
                        t2Ground = getTensor(tf, vertexCoords[2]...)
                        t3Ground = getTensor(tf, vertexCoords[3]...)

                        # If any 2 adjacent tensors are equal, store them both losslessly :(
                        groundTensors = [t1Ground, t2Ground, t3Ground]
                        for pair in ((1,2), (1,3), (2,3))
                            first, second = pair

                            if Matrix{Float32}(groundTensors[first]) == Matrix{Float32}(groundTensors[second]) && (codes[vertexCoords[first]...] ∉ [CODE_LOSSLESS_FULL, CODE_LOSSLESS_FULL_64] || codes[vertexCoords[second]...] ∉ [CODE_LOSSLESS_FULL, CODE_LOSSLESS_FULL_64])
                                storeArray1 = Matrix{Float64}(groundTensors[first])
                                storeArray2 = Matrix{Float64}(groundTensors[second])

                                codes[vertexCoords[first]...] = CODE_LOSSLESS_FULL_64
                                codes[vertexCoords[second]...] = CODE_LOSSLESS_FULL_64

                                setTensor(tf_reconstructed, vertexCoords[first]..., storeArray1)
                                setTensor(tf_reconstructed, vertexCoords[second]..., storeArray2)
                                modified_vertices[first] = true
                                modified_vertices[second] = true
                            end

                        end

                        # If any matrices are equal to 0, store them losslessly
                        for i in 1:3
                            if groundTensors[i] == [0.0 0.0 ; 0.0 0.0]
                                codes[vertexCoords[i]...] = CODE_LOSSLESS_FULL
                                setTensor(tf_reconstructed, vertexCoords[i]..., [0.0 0.0 ; 0.0 0.0])
                            end
                        end

                        t1Recon = getTensor(tf_reconstructed, vertexCoords[1]...)
                        t2Recon = getTensor(tf_reconstructed, vertexCoords[2]...)
                        t3Recon = getTensor(tf_reconstructed, vertexCoords[3]...)

                        if getCircularPointType(t1Ground, t2Ground, t3Ground) != getCircularPointType(t1Recon, t2Recon, t3Recon)
                            for vertex in 1:3
                                coords = vertexCoords[vertex]
                                code = codes[coords...]

                                if code ∉ [CODE_LOSSLESS_FULL, CODE_LOSSLESS_FULL_64, CODE_LOSSLESS_ANGLE] && code % CODE_LOSSLESS_ANGLE != 0
                                    codes[coords...] *= CODE_LOSSLESS_ANGLE

                                    d = d_intermediate[coords...]
                                    r = r_intermediate[coords...]
                                    s = s_intermediate[coords...]
                                    θ = θ_ground[coords...]

                                    θ_intermediate[coords...] = θ_ground[coords...]

                                    d,r,s,θ = adjustDecompositionEntries(d,r,s,θ, ceb, Int64(code))
                                    reconstructedMatrix = recomposeTensor(d,r,s,θ)
                                    setTensor(tf_reconstructed, coords..., reconstructedMatrix)
                                    modified_vertices[vertex] = true
                                end
                            end
                        end

                        if getCircularPointType(t1Ground, t2Ground, t3Ground) != getCircularPointType(getTensor(tf_reconstructed, vertexCoords[1]...), getTensor(tf_reconstructed, vertexCoords[2]...), getTensor(tf_reconstructed, vertexCoords[3]...))
                            for vertex in 1:3
                                if codes[vertexCoords[vertex]...] != CODE_LOSSLESS_FULL_64
                                    codes[vertexCoords[vertex]...] = CODE_LOSSLESS_FULL_64
                                    setTensor(tf_reconstructed, vertexCoords[vertex]..., getTensor(tf, vertexCoords[vertex]...))
                                    modified_vertices[vertex] = true
                                end
                            end
                        end

                        # Check to see if modified points exceed the error bound. If they do, see if storing the angle losslessly
                        # does it, and if not, then store the point losslessly.
                        for newVertex in newVertices

                            coords = vertexCoords[newVertex...]
                            while maximum(abs.(getTensor(tf, coords...) - getTensor(tf_reconstructed, coords...))) > aeb

                                modified_vertices[newVertex] = true
                                code = codes[coords...]
                                if code != CODE_LOSSLESS_FULL

                                    if code % CODE_LOSSLESS_ANGLE != 0

                                        codes[coords...] *= CODE_LOSSLESS_ANGLE

                                        d = d_intermediate[coords...]
                                        r = r_intermediate[coords...]
                                        s = s_intermediate[coords...]
                                        θ = θ_ground[coords...]
    
                                        θ_intermediate[coords...] = θ_ground[coords...]
    
                                        d,r,s,θ = adjustDecompositionEntries(d,r,s,θ, ceb, Int64(code))
                                        reconstructedMatrix = recomposeTensor(d,r,s,θ)
                                        setTensor(tf_reconstructed, coords..., reconstructedMatrix)

                                    else

                                        codes[coords...] = CODE_LOSSLESS_FULL
                                        setTensor(tf_reconstructed, coords..., Matrix{Float32}(getTensor(tf, coords...)))

                                    end

                                else
                                    codes[coords...] = CODE_LOSSLESS_FULL_64
                                    setTensor(tf_reconstructed, coords..., getTensor(tf, coords...))
                                end

                            end

                        end # end of error bound checking

                        # Alter any cells that may have been affected by modified points.
                        # add cells from low to high to minimize the amount of recursion needed.

                        if cell_top

                            if (cell_j, cell_i) < (j_, i_)
                                # add future cells if we are currently in a previous cell.

                                if modified_vertices[3] && (cell_j+1, cell_i+1) < (j_, i_) && cell_i != x
                                    pushIfAbsent!(cellsToVisit, (cell_i+1, cell_j+1, false))
                                end

                                if (modified_vertices[3] || modified_vertices[1]) && (cell_j+1, cell_i) < (j_, i_)
                                    pushIfAbsent!(cellsToVisit, (cell_i, cell_j+1, true))
                                    pushIfAbsent!(cellsToVisit, (cell_i, cell_j+1, false))
                                end
                                
                                if modified_vertices[1] && (cell_j+1, cell_i-1) < (j_, i_) && cell_i != 1
                                    pushIfAbsent!(cellsToVisit, (cell_i-1, cell_j+1, true))
                                    pushIfAbsent!(cellsToVisit, (cell_i-1, cell_j+1, false))
                                end

                                if (modified_vertices[2] || modified_vertices[3]) && (cell_j, cell_i+1) < (j_, i_) && cell_i != x
                                    pushIfAbsent!(cellsToVisit, (cell_i+1, cell_j, true))
                                    pushIfAbsent!(cellsToVisit, (cell_i+1, cell_j, false))
                                end

                            end

                            # add current cell and cell below

                            if (modified_vertices[1] || modified_vertices[2] || modified_vertices[3])
                                pushIfAbsent!(cellsToVisit, (cell_i, cell_j, true))
                            end
                            
                            # add past cells
                            if modified_vertices[1] || modified_vertices[2]
                                pushIfAbsent!(cellsToVisit, (cell_i, cell_j, false))
                            end

                            if modified_vertices[1] && cell_i != 1
                                pushIfAbsent!(cellsToVisit, (cell_i-1, cell_j, true))
                            end

                            if modified_vertices[2] && cell_j != 1
                                if cell_i != x
                                    pushIfAbsent!(cellsToVisit, (cell_i+1, cell_j-1, true))
                                    pushIfAbsent!(cellsToVisit, (cell_i+1, cell_j-1, false))
                                end

                                pushIfAbsent!(cellsToVisit, (cell_i, cell_j-1, true))
                            end

                        else

                            if (cell_j, cell_i) < (j_, i_)
                                if modified_vertices[3] && (cell_j+1,cell_i) < (j_, i_)
                                    pushIfAbsent!(cellsToVisit, (cell_i, cell_j+1, false))
                                end

                                if modified_vertices[3] && (cell_j+1, cell_i-1) < (j_, i_) && cell_i != 1
                                    pushIfAbsent!(cellsToVisit, (cell_i-1, cell_j+1, true))
                                    pushIfAbsent!(cellsToVisit, (cell_i-1, cell_j+1, false))
                                end

                                if modified_vertices[2] && (cell_j, cell_i+1) < (j_, i_) && cell_i != x
                                    pushIfAbsent!(cellsToVisit, (cell_i+1, cell_j, true))
                                end

                                # as long as the main counter is on a higher cell than the current one, 
                                if modified_vertices[2] || modified_vertices[3]
                                    pushIfAbsent!(cellsToVisit, (cell_i, cell_j, true))
                                end

                            end

                            # add current cell
                            if (modified_vertices[1] || modified_vertices[2] || modified_vertices[3])
                                pushIfAbsent!(cellsToVisit, (cell_i, cell_j, false))
                            end

                            # add past cells

                            if (modified_vertices[1] || modified_vertices[3]) && cell_i != 1
                                pushIfAbsent!(cellsToVisit, (cell_i-1, cell_j, true))
                                pushIfAbsent!(cellsToVisit, (cell_i-1, cell_j, false))                                
                            end

                            if modified_vertices[2] && cell_j != 1 && cell_i != x
                                pushIfAbsent!(cellsToVisit, (cell_i+1, cell_j-1, true))
                                pushIfAbsent!(cellsToVisit, (cell_i+1, cell_j-1, false))
                            end

                            if (modified_vertices[1] || modified_vertices[2]) && cell_j != 1
                                pushIfAbsent!(cellsToVisit, (cell_i, cell_j-1, true))
                                pushIfAbsent!(cellsToVisit, (cell_i, cell_j-1, false))
                            end

                            if modified_vertices[1] && cell_i != 1 && cell_j != 1
                                pushIfAbsent!(cellsToVisit, (cell_i-1, cell_j-1, true))
                            end
                        end

                    end
                end
            end 
        end
    end

    numFullLossless = 0
    numLosslessAngle = 0
    
    # Prepare lossless storage
    lossless_storage::Array{Float32} = []
    lossless_storage_64::Array{Float64} = []
    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]

                if codes[t,i,j] == CODE_LOSSLESS_FULL
                    numFullLossless += 1
                    groundTensor = Matrix{Float32}(getTensor(tf, t, i, j))
                    push!(lossless_storage, groundTensor[1,1])
                    push!(lossless_storage, groundTensor[1,2])
                    push!(lossless_storage, groundTensor[2,1])
                    push!(lossless_storage, groundTensor[2,2])
                elseif codes[t,i,j] == CODE_LOSSLESS_FULL_64
                    numFullLossless += 1
                    groundTensor = getTensor(tf, t, i, j)
                    push!(lossless_storage_64, groundTensor[1,1])
                    push!(lossless_storage_64, groundTensor[1,2])
                    push!(lossless_storage_64, groundTensor[2,1])
                    push!(lossless_storage_64, groundTensor[2,2])
                else
                    if codes[t,i,j] % CODE_LOSSLESS_D == 0
                        push!(lossless_storage, d_ground[t,i,j])
                    end

                    if codes[t,i,j] % CODE_LOSSLESS_R == 0
                        push!(lossless_storage, r_ground[t,i,j])
                    end

                    if codes[t,i,j] % CODE_LOSSLESS_S == 0
                        push!(lossless_storage, s_ground[t,i,j])
                    end

                    if codes[t,i,j] % CODE_LOSSLESS_ANGLE == 0
                        numLosslessAngle += 1
                        push!(lossless_storage, θ_ground[t,i,j])
                    end
                end

            end
        end
    end

    # Save metadata and codes

    codesBytes = huffmanEncode(codes)

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])
    write(vals_file, ceb)

    if dtype == Float64
        write(vals_file, 1)
    else
        write(vals_file, 0)
    end

    write(vals_file, length(codesBytes))
    write(vals_file, codesBytes)
    write(vals_file, length(lossless_storage))
    write(vals_file, lossless_storage)
    write(vals_file, length(lossless_storage_64))
    write(vals_file, lossless_storage_64)

    close(vals_file)

    # Compress into a single file

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.xz")

    run(`tar cvf $output_file.tar a.cmp b.cmp c.cmp d.cmp vals.bytes`)
    run(`xz -v9 $output_file.tar`)

    removeIfExists("$output_file.tar")

    cd(cwd)

    # Cleanup all temporary files
    remove("$output/a.dat")
    remove("$output/b.dat")
    remove("$output/c.dat")
    remove("$output/d.dat")

    remove("$output/a.cmp")
    remove("$output/b.cmp")
    remove("$output/c.cmp")
    remove("$output/d.cmp")

    # calculate stats on compressed file size
    # (could be removed in the final version of the program)

    # used for bitrate statistics
    totalTensors = length(codes)

    # shannon entropy
    frequenciesDict = Dict()
    for c in codes
        if haskey(frequenciesDict, c)
            frequenciesDict[c] += 1
        else
            frequenciesDict[c] = 1
        end
    end

    logTotalTensors = log(2,totalTensors)
    sum = 0
    entropy = 0
    for c in keys(frequenciesDict)
        sum += frequenciesDict[c]
        entropy -= (frequenciesDict[c]) * ( log(2, frequenciesDict[c]) - logTotalTensors )
    end
    entropy /= totalTensors

    # bitrate of lossless storage:
    losslessBitrate = (length(lossless_storage) * 32 + length(lossless_storage_64) * 64) / totalTensors

    return (entropy, losslessBitrate)
end

# Error bound is relative
function compress2dSymmetric(containing_folder, dims, output_file, relative_error_bound, output = "../output")
    tf, dtype = loadTensorField2dFromFolder(containing_folder, dims)

    # prepare derived attributes for compression

    r_ground = zeros(Float32, dims)
    θ_ground = zeros(Float32, dims)
    trace_ground = zeros(Float32, dims)

    max_r = -Inf
    max_entry = -Inf
    min_entry = Inf

    for j in 1:dims[3]
        for i in 1:dims[2]
            for t in 1:dims[1]
                matrix = getTensor(tf, t, i, j)

                max_entry = max(max_entry, maximum(matrix))
                min_entry = min(min_entry, minimum(matrix))

                trace_ground[t,i,j] = tr(matrix)

                d = deviator(matrix)
                cplx = d[1,1] + d[1,2]*im

                r = abs(cplx)
                max_r = max(r, max_r)

                r_ground[t,i,j] = r
                θ_ground[t,i,j] = angle(cplx)
            end
        end
    end

    absolute_error_bound = relative_error_bound * (max_entry - min_entry)

    θ_bound = min( absolute_error_bound/(3*max_r), pi/180 )
    remaining_absolute_bound = absolute_error_bound - max_r*θ_bound
    trace_bound = remaining_absolute_bound
    r_bound = remaining_absolute_bound/2

    # Write derived attributes and compress with SZ3

    saveArray("$output/r.dat", r_ground)
    saveArray("$output/theta.dat", θ_ground)
    saveArray("$output/trace.dat", trace_ground)

    run(`../SZ3-master/build/bin/sz3 -f -i $output/theta.dat -z $output/theta.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $θ_bound`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/r.dat -z $output/r.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $r_bound`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/trace.dat -z $output/trace.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $trace_bound`)

    # Decompress and open the compressed files in order to make adjustments

    run(`../SZ3-master/build/bin/sz3 -f -z $output/theta.cmp -o $output/theta_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $θ_bound`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/r.cmp -o $output/r_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $r_bound`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/trace.cmp -o $output/trace_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $trace_bound`)

    tf_reconstructed_entries = reconstructSymmetricEntries2d(dtype, "$output/theta_intermediate.dat", "$output/r_intermediate.dat", "$output/trace_intermediate.dat", dims, r_bound)

    if dtype == Float64
        tf_reconstructed = SymmetricTensorField2d64(tf_reconstructed_entries, dims)
    end

    # Go through the cells and process one at a time, assigning codes to whether or not angles need to be stored losslessly.

    codes = zeros(Float32, dims)
    r_intermediate = loadArray("$output/r_intermediate.dat", Float32, dims)
    trace_intermediate = loadArray("$output/trace_intermediate.dat", Float32, dims)

    pd = ProcessDataSymmetric(tf, tf_reconstructed, θ_ground, r_intermediate, trace_intermediate, r_bound, codes)

    for t in 1:dims[1]
        for i in 1:dims[2]-1
            for j in 1:dims[3]-1
                for k in 0:1
                    process_cell_symmetric(t, i, j, Bool(k), pd)
                end
            end
        end
    end

    # Prepare lossless storage and quantization bytes
    lossless_storage = Array{Float32}(undef, 0)
    codes = vec(pd.codes)
    θ_ground = vec(θ_ground)
    for i in eachindex(codes)
        if codes[i] == CODE_LOSSLESS_ANGLE
            push!(lossless_storage, θ_ground[i])
        end
    end

    # Write header file which contains dimensions

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])

    write(vals_file, θ_bound)
    write(vals_file, r_bound)
    write(vals_file, trace_bound)

    if dtype == Float64
        write(vals_file, 1)
    else
        write(vals_file, 0)
    end

    write(vals_file, length(lossless_storage))
    if length(lossless_storage) != 0
        write(vals_file, lossless_storage)
        write(vals_file, huffmanEncode(codes))
    end

    close(vals_file)

    # Merge compressed files and compress with xz

    cwd = pwd()
    cd(output)

    removeIfExists("$output_file.tar")
    removeIfExists("$output_file.tar.xz")

    run(`tar cvf $output_file.tar r.cmp theta.cmp trace.cmp vals.bytes`)
    run(`xz -v9 $output_file.tar`)

    removeIfExists("$output_file.tar")

    cd(cwd)

    remove("$output/vals.bytes")
    remove("$output/r.cmp")
    remove("$output/theta.cmp")
    remove("$output/trace.cmp")

    remove("$output/r.dat")
    remove("$output/theta.dat")
    remove("$output/trace.dat")

    remove("$output/r_intermediate.dat")
    remove("$output/theta_intermediate.dat")
    remove("$output/trace_intermediate.dat")

end

function process_cell_symmetric(t::Int64, x::Int64, y::Int64, top::Bool, pd::ProcessDataSymmetric)
    ground_type = getCircularPointType(pd.tf_ground, t, x, y, top)
    reconstructed_type = getCircularPointType(pd.tf_reconstructed, t, x, y, top)

    initial_tensors = getTensorsAtCell(pd.tf_reconstructed, t, x, y, top)

    if ground_type != reconstructed_type
        vertices = getCellVertexCoords(t,x,y,top)
        for vertex in vertices
            pd.codes[vertex...] = CODE_LOSSLESS_ANGLE

            r = pd.r_intermediate[vertex...]
            if r <= 0
                r += pd.r_bound
                pd.r_intermediate[vertex...] = r
            end

            cplx = r * exp( pd.θ_ground[vertex...]*im )
            Δ = real(cplx)
            off_diagonal = imag(cplx)
            trace = pd.trace_intermediate[vertex...]
            tensor = [ Δ+0.5*trace off_diagonal ; off_diagonal -Δ+0.5*trace ]
            setTensor(pd.tf_reconstructed, vertex..., tensor)

        end
        
        if top
            process_cell_symmetric(t, x, y, false, pd)
            if x != 1
                process_cell_symmetric(t, x-1, y, true, pd)
            end

            if y != 1
                process_cell_symmetric(t, x, y-1, true, pd)
            end
        else
            if x != 1
                process_cell_symmetric(t, x-1, y, true, pd)
                process_cell_symmetric(t, x-1, y, false, pd)
            end

            if y != 1
                process_cell_symmetric(t, x, y-1, true, pd)
                process_cell_symmetric(t, x, y-1, false, pd)
            end

            if x != 1 && y != 1
                process_cell_symmetric(t, x-1, y-1, true, pd)
            end
        end
    end

    ground_type = getCircularPointType(pd.tf_ground, t, x, y, top)
    reconstructed_type = getCircularPointType(pd.tf_reconstructed, t, x, y, top)
    if ground_type != reconstructed_type
        println("something went wrong $t $x $y")

        ground_tensors = getTensorsAtCell(pd.tf_ground, t,x,y,top)
        recon_tensors = getTensorsAtCell(pd.tf_reconstructed, t,x,y,top)

        println("ground type: $ground_type")
        for t in ground_tensors
            println(t)
        end

        println("reconstructed type: $reconstructed_type")
        for t in recon_tensors
            println(t)
        end

        println("---")
        vertices = getCellVertexCoords(t,x,y,top)
        for v in vertices
            println("v: $v")
            println(pd.r_intermediate[v...])
            println(pd.trace_intermediate[v...])
            println(pd.θ_ground[v...])
            println("---")
        end

        exit()
    end    

end

end