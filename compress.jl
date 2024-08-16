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
    precision::Array{Int64}
    precisionStage::Array{Int64} # precision goes through a full set twice, the second time with a lossless angle. This keeps track of where in that process the precision is.
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
function initialProcessPoint( pd::ProcessData, t::Int64, x::Int64, y::Int64 )
    if pd.codes[t,x,y] == 0
        processPoint( pd, t, x, y )
        return true
    end
    return false
end

function raisePrecisionAndProcess( pd::ProcessData, t::Int64, x::Int64, y::Int64 )
    angleIsLossless = raisePrecision( pd, t, x, y, pd.codes[t,x,y]%CODE_LOSSLESS_ANGLE==0 )
    processPoint( pd, t, x, y, true, angleIsLossless )

end

# Does not lower precisions. Keeps precisions at the same spot but also stores the angles losslessly.
# This is to not disturb the isocontours used in producing the eigenvector graph.
function storeAngleLosslesslyAndProcess( pd::ProcessData, t::Int64, x::Int64, y::Int64 )

    if pd.codes[t,x,y]%CODE_LOSSLESS_ANGLE != 0
        pd.codes[t,x,y] *= CODE_LOSSLESS_ANGLE
        pd.precision[t,x,y] = 0
        processPoint( pd, t, x, y, true, true )
    end

end

# passing if the angle is lossless helps prevent needing to modulo a million times
# returns true if the angle is stored losslessly and false otherwise.
function raisePrecision( pd::ProcessData, t::Int64, x::Int64, y::Int64, angleIsLossless::Bool )
    pd.precisionStage[t,x,y] += 1
    pd.precision[t,x,y] += 1

    if pd.precision[t,x,y] > MAX_PRECISION

        pd.precision[t,x,y] = 0

        if angleIsLossless
            pd.codes[t,x,y] = CODE_LOSSLESS_FULL_64
            setTensor(pd.tf_reconstructed, t, x, y, getTensor(pd.tf, t, x, y))

            pd.d_quantization[t,x,y] = 0
            pd.r_quantization[t,x,y] = 0
            pd.s_quantization[t,x,y] = 0

        else
            pd.codes[t,x,y] *= CODE_LOSSLESS_ANGLE
        end

        return true
    else
        return angleIsLossless
    end
end

# passedAngleIsLossless is set to true if we are passing the angleIsLossless parameter
function processPoint( pd::ProcessData, t::Int64, x::Int64, y::Int64, passedAngleIsLossless=false, angleIsLossless=false )

    code = pd.codes[t,x,y]

    if !passedAngleIsLossless
        angleIsLossless = (code != 0) && (code % CODE_LOSSLESS_ANGLE == 0)
    end

    if code != CODE_LOSSLESS_FULL && code != CODE_LOSSLESS_FULL_64
        while !processPointDRS( pd, t, x, y, angleIsLossless )
            angleIsLossless = raisePrecision( pd, t, x, y, angleIsLossless )
        end
    end

end

# process the d, r, and s values of a point
function processPointDRS( pd::ProcessData, t::Int64, x::Int64, y::Int64, losslessAngle::Bool )
    code = pd.codes[t,x,y]
    if code == CODE_LOSSLESS_FULL || code == CODE_LOSSLESS_FULL_64
        return true
    end

    groundTensor = getTensor(pd.tf, t, x, y)

    s_intermediate = pd.s_intermediate[t,x,y]
    if s_intermediate < 0
        s_intermediate += pd.aeb
    end

    pd.d_quantization[t,x,y] = round( (pd.d_ground[t,x,y] - pd.d_intermediate[t,x,y]) * 2^(pd.precision[t,x,y]) / (2*pd.aeb) )
    pd.r_quantization[t,x,y] = round( (pd.r_ground[t,x,y] - pd.r_intermediate[t,x,y]) * 2^(pd.precision[t,x,y]) / (2*pd.aeb) )
    pd.s_quantization[t,x,y] = round( (pd.s_ground[t,x,y] - s_intermediate) * 2^(pd.precision[t,x,y]) / (2*pd.aeb) )

    d_ = pd.d_intermediate[t, x, y] + 2 * pd.aeb * pd.d_quantization[t,x,y] / (2^(pd.precision[t,x,y]))
    r_ = pd.r_intermediate[t, x, y] + 2 * pd.aeb * pd.r_quantization[t,x,y] / (2^(pd.precision[t,x,y]))
    s_ = s_intermediate + 2 * pd.aeb * pd.s_quantization[t,x,y] / (2^(pd.precision[t,x,y]))

    while s_ < 0
        pd.s_quantization[t,x,y] += 1
        s_ = s_intermediate + 2 * pd.aeb * pd.s_quantization[t,x,y] / (2^(pd.precision[t,x,y]))        
    end

    if losslessAngle
        θ_ = pd.θ_ground[t, x, y]
        code = CODE_LOSSLESS_ANGLE
    else
        θ_ = pd.θ_intermediate[t, x, y]
        code = 1
    end

    # Use these manipulations to avoid numerical precision issues
    d_ground64, r_ground64, s_ground64, θ_ground64 = decomposeTensor(groundTensor)

    reconstructedMatrix = recomposeTensor(d_, r_, s_, θ_)
    d_recompose, r_recompose, s_recompose, θ_recompose = decomposeTensor(reconstructedMatrix)

    eVectorGround = classifyTensorEigenvector(r_ground64, s_ground64)
    eVectorIntermediate = classifyTensorEigenvector(r_recompose, s_recompose)
    
    eValueGround = classifyTensorEigenvalue(d_ground64, r_ground64, s_ground64)
    eValueIntermediate = classifyTensorEigenvalue(d_recompose, r_recompose, s_recompose)

    worked = true

    # Modify vertex if either classification doesn't match
    if eVectorGround != eVectorIntermediate || eValueGround != eValueIntermediate

        if eVectorGround != eVectorIntermediate
            code *= CODE_CHANGE_EIGENVECTOR^eVectorGround
        end

        if eValueGround != eValueIntermediate
            code *= CODE_CHANGE_EIGENVALUE^eValueGround
        end

        d_ground32 = pd.d_ground[t, x, y]
        r_ground32 = pd.r_ground[t, x, y]
        s_ground32 = pd.s_ground[t, x, y]

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

        d_, r_, s_, θ_ = adjustDecompositionEntries(d_, r_, s_, θ_, pd.aeb, code)
        reconstructedMatrix = recomposeTensor(d_, r_, s_, θ_)
        d_, r_, s_, θ_ = decomposeTensor(reconstructedMatrix)

        # Check if these adjustments worked. If not, store all entries losslessly.
        if classifyTensorEigenvector(r_, s_) != eVectorGround || classifyTensorEigenvalue(d_, r_, s_) != eValueGround
            worked = false
        end

    end

    pd.codes[t,x,y] = code
    setTensor(pd.tf_reconstructed, t, x, y, reconstructedMatrix)

    if worked && maximum( abs.( reconstructedMatrix - groundTensor ) ) > pd.aeb
        worked = false
    end

    return worked

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

    saveArray("$output/a.dat", A_ground)
    saveArray("$output/b.dat", B_ground)
    saveArray("$output/c.dat", C_ground)
    saveArray("$output/d.dat", D_ground)

    # Initial compression

    run(`../SZ3-master/build/bin/sz3 -f -i $output/a.dat -z $output/a.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/b.dat -z $output/b.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/c.dat -z $output/c.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -i $output/d.dat -z $output/d.cmp -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)

    # Initial decompression

    run(`../SZ3-master/build/bin/sz3 -f -z $output/a.cmp -o $output/a_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/b.cmp -o $output/b_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/c.cmp -o $output/c_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    run(`../SZ3-master/build/bin/sz3 -f -z $output/d.cmp -o $output/d_intermediate.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)

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

    pd = ProcessData(
        aeb,
        tf,
        tf_reconstructed,
        d_ground,
        r_ground,
        s_ground,
        θ_ground,
        d_intermediate,
        r_intermediate,
        s_intermediate,
        θ_intermediate,
        zeros(Int64, dims),
        zeros(Int64, dims),
        zeros(Int64, dims),
        zeros(Int64, dims),
        zeros(Int64, dims),
        codes
    )

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
                                newVertices = [3]
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
                            if pd.codes[t, vertex_i, vertex_j] == 0

                                modified_vertices[newVertex] = true
                                initialProcessPoint( pd, t, vertex_i, vertex_j )

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

                                while class1 != class2
                                    keepChecking = true
                                    modified_vertices[edge[1]] = true
                                    modified_vertices[edge[2]] = true

                                    # also using "d" here since the precisions are all lock-step. But could
                                    # unlock them later maybe idk.
                                    lossless1 = pd.codes[t, i1, j1] in [CODE_LOSSLESS_FULL, CODE_LOSSLESS_FULL_64]
                                    lossless2 = pd.codes[t, i2, j2] in [CODE_LOSSLESS_FULL, CODE_LOSSLESS_FULL_64]
                                    precision1 = pd.precisionStage[t, i1, j1]
                                    precision2 = pd.precisionStage[t, i2, j2]

                                    if lossless1 && lossless2
                                        println("something is wrong")
                                        exit()
                                    end

                                    if precision1 < precision2 && !lossless1
                                        raisePrecisionAndProcess( pd, t, i1, j1 )
                                    elseif precision1 > precision2 && !lossless2
                                        raisePrecisionAndProcess( pd, t, i2, j2 )
                                    else
                                        raisePrecisionAndProcess( pd, t, i1, j1 )
                                        raisePrecisionAndProcess( pd, t, i2, j2 )
                                    end

                                    t12 = getTensor(tf_reconstructed, t, i1, j1)       
                                    t22 = getTensor(tf_reconstructed, t, i2, j2)
                                    class2 = classifyEdgeEigenvalue(t12, t22)
                                    
                                end

                            end

                        end

                        # cell

                        # # If any 2 adjacent tensors are equal, store them both losslessly :(
                        # groundTensors = [t1Ground, t2Ground, t3Ground]
                        # for pair in ((1,2), (1,3), (2,3))
                        #     first, second = pair

                        #     if Matrix{Float32}(groundTensors[first]) == Matrix{Float32}(groundTensors[second]) && (codes[vertexCoords[first]...] ∉ [CODE_LOSSLESS_FULL, CODE_LOSSLESS_FULL_64] || codes[vertexCoords[second]...] ∉ [CODE_LOSSLESS_FULL, CODE_LOSSLESS_FULL_64])
                        #         storeArray1 = Matrix{Float64}(groundTensors[first])
                        #         storeArray2 = Matrix{Float64}(groundTensors[second])

                        #         codes[vertexCoords[first]...] = CODE_LOSSLESS_FULL_64
                        #         codes[vertexCoords[second]...] = CODE_LOSSLESS_FULL_64

                        #         setTensor(tf_reconstructed, vertexCoords[first]..., storeArray1)
                        #         setTensor(tf_reconstructed, vertexCoords[second]..., storeArray2)
                        #         modified_vertices[first] = true
                        #         modified_vertices[second] = true
                        #     end

                        # end

                        # # If any matrices are equal to 0, store them losslessly
                        # for i in 1:3
                        #     if groundTensors[i] == [0.0 0.0 ; 0.0 0.0]
                        #         codes[vertexCoords[i]...] = CODE_LOSSLESS_FULL
                        #         setTensor(tf_reconstructed, vertexCoords[i]..., [0.0 0.0 ; 0.0 0.0])
                        #     end
                        # end

                        t1Ground = getTensor(tf, vertexCoords[1]...)
                        t2Ground = getTensor(tf, vertexCoords[2]...)
                        t3Ground = getTensor(tf, vertexCoords[3]...)

                        t1Recon = getTensor(tf_reconstructed, vertexCoords[1]...)
                        t2Recon = getTensor(tf_reconstructed, vertexCoords[2]...)
                        t3Recon = getTensor(tf_reconstructed, vertexCoords[3]...)

                        groundCellType = getCircularPointType(t1Ground, t2Ground, t3Ground)
                        reconCellType = getCircularPointType(t1Recon, t2Recon, t3Recon)

                        while groundCellType != reconCellType

                            ps1 = pd.precisionStage[vertexCoords[1]...]
                            ps2 = pd.precisionStage[vertexCoords[2]...]
                            ps3 = pd.precisionStage[vertexCoords[3]...]

                            minPrecisionStage = min(ps1, ps2, ps3)

                            if ps1 == minPrecisionStage
                                raisePrecisionAndProcess(pd, vertexCoords[1]...)
                            end

                            if ps2 == minPrecisionStage
                                raisePrecisionAndProcess(pd, vertexCoords[2]...)
                            end

                            if ps3 == minPrecisionStage
                                raisePrecisionAndProcess(pd, vertexCoords[3]...)
                            end

                            t1Recon = getTensor(tf_reconstructed, vertexCoords[1]...)
                            t2Recon = getTensor(tf_reconstructed, vertexCoords[2]...)
                            t3Recon = getTensor(tf_reconstructed, vertexCoords[3]...)

                            reconCellType = getCircularPointType(t1Recon, t2Recon, t3Recon)

                            modified_vertices[1] = true
                            modified_vertices[2] = true
                            modified_vertices[3] = true

                        end

                        # if getCircularPointType(t1Ground, t2Ground, t3Ground) != getCircularPointType(getTensor(tf_reconstructed, vertexCoords[1]...), getTensor(tf_reconstructed, vertexCoords[2]...), getTensor(tf_reconstructed, vertexCoords[3]...))
                        #     for vertex in 1:3
                        #         if codes[vertexCoords[vertex]...] != CODE_LOSSLESS_FULL_64
                        #             codes[vertexCoords[vertex]...] = CODE_LOSSLESS_FULL_64
                        #             setTensor(tf_reconstructed, vertexCoords[vertex]..., getTensor(tf, vertexCoords[vertex]...))
                        #             modified_vertices[vertex] = true
                        #         end
                        #     end
                        # end

                        # # Check to see if modified points exceed the error bound. If they do, see if storing the angle losslessly
                        # # does it, and if not, then store the point losslessly.
                        # for newVertex in newVertices

                        #     coords = vertexCoords[newVertex...]
                        #     while maximum(abs.(getTensor(tf, coords...) - getTensor(tf_reconstructed, coords...))) > aeb

                        #         modified_vertices[newVertex] = true
                        #         code = codes[coords...]
                        #         if code != CODE_LOSSLESS_FULL

                        #             if code % CODE_LOSSLESS_ANGLE != 0

                        #                 codes[coords...] *= CODE_LOSSLESS_ANGLE

                        #                 d = d_intermediate[coords...]
                        #                 r = r_intermediate[coords...]
                        #                 s = s_intermediate[coords...]
                        #                 θ = θ_ground[coords...]
    
                        #                 θ_intermediate[coords...] = θ_ground[coords...]
    
                        #                 d,r,s,θ = adjustDecompositionEntries(d,r,s,θ, ceb, Int64(code))
                        #                 reconstructedMatrix = recomposeTensor(d,r,s,θ)
                        #                 setTensor(tf_reconstructed, coords..., reconstructedMatrix)

                        #             else

                        #                 codes[coords...] = CODE_LOSSLESS_FULL
                        #                 setTensor(tf_reconstructed, coords..., Matrix{Float32}(getTensor(tf, coords...)))

                        #             end

                        #         else
                        #             codes[coords...] = CODE_LOSSLESS_FULL_64
                        #             setTensor(tf_reconstructed, coords..., getTensor(tf, coords...))
                        #         end

                        #     end

                        # end # end of error bound checking

                        # # Alter any cells that may have been affected by modified points.
                        # # add cells from low to high to minimize the amount of recursion needed.

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
    codes = pd.codes
    
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

    # precisions:
    d_quantization_final = vec(pd.d_quantization .* ( 2 .^ (MAX_PRECISION .- pd.precision) ))
    r_quantization_final = vec(pd.r_quantization .* ( 2 .^ (MAX_PRECISION .- pd.precision) ))
    s_quantization_final = vec(pd.s_quantization .* ( 2 .^ (MAX_PRECISION .- pd.precision) ))
    quantization_codes = vcat(d_quantization_final, r_quantization_final, s_quantization_final)

    # Save metadata and codes

    codesBytes = huffmanEncode(codes)
    quantizationBytes = huffmanEncode(quantization_codes)

    vals_file = open("$output/vals.bytes", "w")
    write(vals_file, dims[1])
    write(vals_file, dims[2])
    write(vals_file, dims[3])
    write(vals_file, aeb)

    if dtype == Float64
        write(vals_file, 1)
    else
        write(vals_file, 0)
    end

    write(vals_file, length(codesBytes))
    write(vals_file, codesBytes)
    write(vals_file, length(quantizationBytes))
    write(vals_file, quantizationBytes)
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

    entropy2 = 0
    sum2 = 0

    frequenciesDict2 = Dict()
    for c in quantization_codes
        if haskey(frequenciesDict2, c)
            frequenciesDict2[c] += 1
        else
            frequenciesDict2[c] = 1
        end
    end

    for c in keys(frequenciesDict2)
        sum2 += frequenciesDict2[c]
        entropy2 -= (frequenciesDict2[c]) * ( log(2, frequenciesDict2[c]) - logTotalTensors - log(2,3) )
    end

    entropy2 /= totalTensors

    entropy += entropy2

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