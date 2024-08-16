module plotTensorField

using Plots
using LinearAlgebra

using ..tensorField
using ..utils

export plotEigenFieldGlyphs2d
export plotDegenerateLines
export plotEigenvectorGraph
export plotEigenvalueGraph

function plotShape(x, y, color, show_borders = false, linewidth=1)
    if show_borders
        return plot!(Shape(x, y), fill=color, legend=false)
    else
        return plot!(Shape(x, y), fill=color, linecolor=color, linewidth=linewidth, legend=false)        
    end
end

# which should take the value of 1 or 2
function plotEigenFieldGlyphs2d(tf::TensorField2d, t::Int64, which::Int64, slice::Tuple{Int64, Int64, Int64, Int64} = (-1,-1,-1,-1), color="black")

    if slice == (-1,-1,-1,-1)
        xMax, yMax = tf.dims[2], tf.dims[3]
        xMin = 1
        yMin = 1
    else
        xMin, xMax, yMin, yMax = slice
    end

    vecs = []
    p = plot([], [], legend=false, color=color)

    for i in xMin:xMax
        for j in yMin:yMax
            matrix = getTensor(tf, t, i, j)
            e_vecs = eigvecs(matrix)
            v = real.(e_vecs[:,which])

            push!( vecs,   ( (i-0.5*v[1], i+0.5*v[1]) , (j-0.5*v[2],j+0.5*v[2]) )   )
            p = plot!([i-0.5*v[1],i+0.5*v[1]], [j-0.5*v[2],j+0.5*v[2]], legend=false, color=color)

        end
    end

    # Plot critical points
    x -= 1
    y -= 1
    for i in xMin:xMax
        for j in yMin:yMax
            for k in 0:1
                cp_type = getCircularPointType(tf, t, i, j, Bool(k))
                if cp_type == CP_TRISECTOR || cp_type == CP_WEDGE
                    if k == 0
                        cp_x = i+0.25
                        cp_y = j+0.25
                    else
                        cp_x = i+0.75
                        cp_y = i+0.75
                    end

                    if cp_type == CP_TRISECTOR
                        p = scatter!([cp_x], [cp_y], color="white")
                    else
                        p = scatter!([cp_x], [cp_y], color="black")
                    end 
                end
            end
        end
    end

    return p

end

function plotEigenvectorGraph(tf::TensorField2d, t::Int64, slice::Tuple{Int64, Int64, Int64, Int64}, show_borders=false, show_points = false)

    if slice == (-1,-1,-1,-1)
        xMax, yMax = tf.dims[2]-1, tf.dims[3]-1
        xMin = 1
        yMin = 1
    else
        xMin, xMax, yMin, yMax = slice
        xMax -= 1
        yMax -= 1
    end

    p = plot([], [], legend=false, color="black")

    # array storing the (x1, x2, y1, y2, colorid) of each line segment passing through a cell as a border
    # we do it last so that the cell borders don't paint over these segment borders at all.
    borderLines::Array{Tuple{Float64, Float64, Float64, Float64, Int64}} = []

    # array storing (x,y,color) of each individual point that we will plot.
    scatterPoints::Array{Tuple{Float64, Float64, String}} = []

    for i in xMin:xMax
        for j in yMin:yMax
            for k in 0:1

                # Set up 
                
                positions = getCellVertexCoords(t, i, j, Bool(k))
                tensors = getTensorsAtCell(tf, t, i, j, Bool(k))

                decompositions::Array{Tuple{Float64, Float64, Float64, Float64}} = [(0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0)]
                classifications = [-1, -1, -1]

                # Classify the three tensors in the cell
                for i in 1:3
                    decomp = decomposeTensor(tensors[i])
                    decompositions[i] = decomp

                    _, yr, ys, _ = decomp
                    if abs(yr) < ϵ
                        classifications[i] = SYMMETRIC
                    elseif yr > 0
                        if abs(yr - ys) < ϵ
                            classifications[i] = PI_BY_4
                        elseif yr > ys
                            classifications[i] = W_CN
                        else
                            classifications[i] = W_RN
                        end
                    else
                        if abs(yr + ys) < ϵ
                            classifications[i] = MINUS_PI_BY_4                        
                        elseif -yr > ys
                            classifications[i] = W_CS
                        else
                            classifications[i] = W_RS
                        end
                    end
                end

                # Depending on which tensors are equal, plot differently

                if classifications[1] == classifications[2] == classifications[3]
                    # if the 3 tensors have the same type, just fill in the color with the color corresponding to that type
                    #p = plot!(Shape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]]), fill=eigenvectorColors[classifications[1]])
                    p = plotShape([ positions[1][2], positions[2][2], positions[3][2] ], [positions[1][3], positions[2][3], positions[3][3]], eigenvectorColors[classifications[1]], show_borders)
                else
                    # if they are not all equal

                    # First, set the middle vertex as the "anchor" for which other vertices will come out from.
                    if classifications[1] <= classifications[2] <= classifications[3] || classifications[3] <= classifications[2] <= classifications[1]
                        anchor = 2
                        other_vertices = [1,3]
                        other_vertices_dual = [3,1]
                    elseif classifications[2] <= classifications[1] <= classifications[3] || classifications[3] <= classifications[1] <= classifications[2]
                        anchor = 1
                        other_vertices = [2,3]
                        other_vertices_dual = [3,2]
                    else
                        anchor = 3
                        other_vertices = [1,2]
                        other_vertices_dual = [2,1]
                    end

                    anchor_classification = classifications[anchor]

                    # Color the base color of the triangle
                    if anchor_classification == SYMMETRIC || anchor_classification == PI_BY_4 || anchor_classification == MINUS_PI_BY_4
                        # Handle an edge case where one of the edges is a border color
                        # this can only happen if 2 vertices have the same classification which is itself a border color.
                        # Color that edge as the border and fill in the rest of the triangle to be off by 1 from there
                        if anchor_classification == classifications[other_vertices[1]]
                            double = other_vertices[1]
                            single = other_vertices[2]
                        elseif anchor_classification == classifications[other_vertices[2]]
                            double = other_vertices[2]
                            single = other_vertices[1]
                        else
                            double = -1
                        end

                        if double != -1
                            if anchor_classification > classifications[single]
                                #plot!(Shape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]]), fill=eigenvectorColors[anchor_classification-1])
                                plotShape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]], eigenvectorColors[anchor_classification-1], show_borders)
                            else
                                #plot!(Shape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]]), fill=eigenvectorColors[anchor_classification+1])
                                plotShape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]], eigenvectorColors[anchor_classification+1], show_borders)
                            end
                            
                            # Then plot the line that goes across the edge
                            push!(borderLines, (positions[double][2], positions[anchor][2], positions[double][3], positions[anchor][3], anchor_classification))
                        end
                    else
                        # Otherwise, fill in the triangle with the color of the anchor
                        #p = plot!(Shape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]]), fill=eigenvectorColors[anchor_classification])
                        plotShape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]], eigenvectorColors[anchor_classification], show_borders)
                    end

                    # For the 2 vertices that are not the anchor, draw any dividing lines between them as needed
                    for other_index in 1:2

                        other_vertex = other_vertices[other_index]
                        other_classification = classifications[other_vertex]
                        interp_vertex_indices = [anchor, other_vertices_dual[other_index]]                        

                        if other_classification == anchor_classification
                            continue
                        end

                        # check if numbers are increasing or decreasing as you travel from anchor -> other_vertex
                        if anchor_classification > other_classification
                            direction = -1
                        else
                            direction = 1
                        end

                        # Store interpolation points along each edge that would correspond to the pi/4 and -pi/4 latitutde lines
                        solutions_pi_by_4 = [-1.0,-1.0]
                        solutions_minus_pi_by_4 = [-1.0,-1.0]

                        # If our cell has any real->complex transitions, then figure out where the correct intercepts are.
                        if direction*anchor_classification < direction*PI_BY_4 < direction*other_classification || direction*anchor_classification < direction*MINUS_PI_BY_4 < direction*other_classification

                            t1 = tensors[other_vertex]

                            # This sets up a solution to a quadratic equation that we are given to find the proper intersection points with the cell.
                            k11 = t1[1,1] - t1[2,2]
                            k12 = t1[1,2] + t1[2,1]
                            k13 = t1[2,1] - t1[1,2]

                            for ivi in 1:2
                                t2 = tensors[interp_vertex_indices[ivi]]

                                k21 = t2[1,1] - t2[2,2]
                                k22 = t2[1,2] + t2[2,1]
                                k23 = t2[2,1] - t2[1,2]

                                c1 = (k11 - k21)^2 + (k12 - k22)^2 - (k13 - k23)^2
                                c2 = 2*( k21*(k11-k21) + k22*(k12-k22) - k23*(k13-k23) )
                                c3 = k21^2 + k22^2 - k23^2

                                s1 = (-c2 + sqrt(c2^2-4*c1*c3))/(2*c1)
                                s2 = (-c2 - sqrt(c2^2-4*c1*c3))/(2*c1)

                                if direction*anchor_classification < direction*PI_BY_4 < direction*other_classification
                                    if direction*anchor_classification < direction*MINUS_PI_BY_4 < direction*other_classification
                                        if direction == 1
                                            solutions_pi_by_4[ivi] = min(s1, s2)
                                            solutions_minus_pi_by_4[ivi] = max(s1, s2)
                                        else
                                            solutions_pi_by_4[ivi] = max(s1,s2)
                                            solutions_minus_pi_by_4[ivi] = min(s1,s2)
                                        end
                                    else
                                        if 0 <= s1 <= 1
                                            solutions_pi_by_4[ivi] = s1
                                        else
                                            solutions_pi_by_4[ivi] = s2
                                        end
                                    end
                                else
                                    if 0 <= s1 <= 1
                                        solutions_minus_pi_by_4[ivi] = s1
                                    else
                                        solutions_minus_pi_by_4[ivi] = s2
                                    end
                                end

                            end
                        end

                        solutions_symmetric = [-1.0, -1.0]

                        # if our cell has any symmetric tensors running through it, figure out where that line should go as well.
                        if direction*anchor_classification < direction*SYMMETRIC < direction*other_classification
                            t1 = tensors[other_vertex]
                            yr1 = t1[2,1]-t1[1,2]

                            for ivi in 1:2
                                t2 = tensors[interp_vertex_indices[ivi]]
                                yr2 = t2[2,1]-t2[1,2]

                                solutions_symmetric[ivi] = yr2/(yr2-yr1)
                            end
                        end

                        # Now plot the relevant triangles

                        if direction == 1
                            solutions = (solutions_pi_by_4, solutions_symmetric, solutions_minus_pi_by_4)
                            line_colors = (PI_BY_4, SYMMETRIC, MINUS_PI_BY_4)
                            block_colors = (W_RN, W_RS, W_CS)
                        else
                            solutions = (solutions_minus_pi_by_4, solutions_symmetric, solutions_pi_by_4)
                            line_colors = (MINUS_PI_BY_4, SYMMETRIC, PI_BY_4)
                            block_colors = (W_RS, W_RN, W_CN)
                        end

                        _, vertex_x, vertex_y = positions[other_vertex]
                        _, double1_x, double1_y = positions[interp_vertex_indices[1]]
                        _, double2_x, double2_y = positions[interp_vertex_indices[2]]

                        for division in 1:3
                            interp1, interp2 = solutions[division]

                            if 0 <= interp1 <= 1
                                border1_x = interp1*vertex_x + (1-interp1)*double1_x
                                border1_y = interp1*vertex_y + (1-interp1)*double1_y

                                border2_x = interp2*vertex_x + (1-interp2)*double2_x
                                border2_y = interp2*vertex_y + (1-interp2)*double2_y

                                #plot!(Shape([border1_x, vertex_x, border2_x], [border1_y, vertex_y, border2_y]), fill=eigenvectorColors[block_colors[division]])
                                plotShape([border1_x, vertex_x, border2_x], [border1_y, vertex_y, border2_y], eigenvectorColors[block_colors[division]], show_borders)
                                push!(borderLines, (border1_x, border2_x, border1_y, border2_y, line_colors[division]))
                            end
                        end
                    end
                end
 
                # Plot individual points or ellipses that are contained within cells


                # Plot the interior elliptical region corresponding to a move between real and complex (if it exists)
                if !( minimum(classifications) < PI_BY_4 < maximum(classifications) || minimum(classifications) < MINUS_PI_BY_4 < maximum(classifications) )

                    t1, t2, t3 = tensors

                    C = [ t1[1,1]-t1[2,2] t2[1,1]-t2[2,2] t3[1,1]-t3[2,2] ; 
                        t1[1,2]+t1[2,1] t2[1,2]+t2[2,1] t3[1,2]+t3[2,1] ; 
                        t1[2,1]-t1[1,2] t2[2,1]-t2[1,2] t3[2,1]-t3[1,2] ]
                    
                    T = [ positions[1][2] positions[2][2] positions[3][2] ;
                        positions[1][3] positions[2][3] positions[3][3] ;
                        1               1               1               ]

                    M = T^-1
                    
                    K = C*M

                    # Coefficients of conic section f(x,y) = Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
                    # This conic is where yr = ys

                    A = K[1,1]^2 + K[2,1]^2 - K[3,1]^2
                    B = 2*(K[1,1]*K[1,2] + K[2,1]*K[2,2] - K[3,1]*K[3,2])
                    C = K[1,2]^2 + K[2,2]^2 - K[3,2]^2
                    D = 2*(K[1,1]*K[1,3] + K[2,1]*K[2,3] - K[3,1]*K[3,3])
                    E = 2*(K[1,2]*K[1,3] + K[2,2]*K[2,3] - K[3,2]*K[3,3])
                    F = K[1,3]^2 + K[2,3]^2 - K[3,3]^2

                    # By making sure we have positive coefficients, f(x,y) > 0 we are outside the ellipse.
                    if A < 0
                        A *= -1
                        B *= -1
                        C *= -1
                        D *= -1
                        E *= -1
                        F *= -1
                    end

                    discriminant = B^2 - 4*A*C

                    # Check if the conic is indeed an ellipse. If not, we don't have an ellipse inside of the cell obviously.
                    if discriminant < 0

                        # Check if a vertex is contained within the ellipse
                        _, vx, vy = positions[1]
                        ellipse_value = A*vx^2 + B*vx*vy + C*vy^2 + D*vx + E*vy + F

                        # In this case, the vertex is not within the ellipse meaning that the ellipse is either entirely
                        # inside or outside the cell
                        if ellipse_value > 0
                            # Check to see if the center is inside of the cell.
                            center = [ 2*C*D-B*E ; 2*A*E-B*D ; discriminant ] / discriminant
                            barrycenter_vals = M*center
                            if 0 <= barrycenter_vals[1] <= 1 && 0 <= barrycenter_vals[2] <= 1 && 0 <= barrycenter_vals[3] <= 1
                                # If the center is in the cell, then plot the ellipse as a dot.
                                center_tensor = barrycenter_vals[1]*t1 + barrycenter_vals[2]*t2 + barrycenter_vals[3]*t3
                                center_classification = classifyTensorEigenvector(center_tensor)
                                push!(scatterPoints, (center[1], center[2], eigenvectorColors[center_classification]))
                            end
                        end
                    end
                end

                # Plot circular points
                _, _, ys1, θ_1 = decompositions[1]
                _, _, ys2, θ_2 = decompositions[2]
                _, _, ys3, θ_3 = decompositions[3]

                sym1 = ys1 * [ cos(θ_1) sin(θ_1) ; sin(θ_1) -cos(θ_1) ]
                sym2 = ys2 * [ cos(θ_2) sin(θ_2) ; sin(θ_2) -cos(θ_2) ]
                sym3 = ys3 * [ cos(θ_3) sin(θ_3) ; sin(θ_3) -cos(θ_3) ]

                circularPointType = getCircularPointType(sym1, sym2, sym3)

                if circularPointType == CP_TRISECTOR || circularPointType == CP_WEDGE
                    center = ( (positions[1][2] + positions[2][2] + positions[3][2])/3, (positions[1][3] + positions[2][3] + positions[3][3])/3 )

                    if circularPointType == CP_TRISECTOR
                        push!(scatterPoints, (center[1], center[2], "white"))
                    else
                        push!(scatterPoints, (center[1], center[2], "black"))
                    end
                end

            end
        end
    end

    for tup in borderLines
        p = plot!( [tup[1], tup[2]], [tup[3], tup[4]], color=eigenvectorColors[tup[5]] )
    end

    if show_points
        for tup in scatterPoints
            p = scatter!( [tup[1]], [tup[2]], color=tup[3], markersize=2 )
        end
    end

    return p

end

function plotEigenvalueGraph(tf::TensorField2d, t::Int64, slice::Tuple{Int64, Int64, Int64, Int64} = (-1,-1,-1,-1), show_borders = false, show_points = false)

    if slice == (-1,-1,-1,-1)
        xMax, yMax = tf.dims[2]-1, tf.dims[3]-1
        xMin = 1
        yMin = 1
    else
        xMin, xMax, yMin, yMax = slice
        xMax -= 1
        yMax -= 1
    end

    p = plot([], [])

    # List of [x0, x1, y0, y1] for drawing borders.
    borderLines1::Array{Tuple{Float64, Float64, Float64, Float64}} = []
    borderLines2::Array{Tuple{Float64, Float64, Float64, Float64}} = []
    anisotropicShapes::Array{Shape{Float64, Float64}} = []

    # for i={1,2,3}, we say that edge i has the other 2 as vertices e.g. edge 2 has vertices 1 and 3
    edges = [ 2 3 ; 3 1 ; 1 2 ]

    for i in xMin:xMax
        for j in yMin:yMax
            for k in 0:1

                positions = getCellVertexCoords(t, i, j, Bool(k))
                tensors = getTensorsAtCell(tf, t, i, j, Bool(k))

                decompositions::Array{Tuple{Float64, Float64, Float64, Float64}} = [(0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0)]
                classifications = [-1, -1, -1]

                # Classify the three tensors in the cell
                for i in 1:3
                    decomp = decomposeTensor(tensors[i])
                    decompositions[i] = decomp

                    yd, yr, ys, _ = decomp
                    if ys > abs(yd) && ys > abs(yr)
                        classifications[i] = ANISOTROPIC_STRETCHING

                    elseif abs(yr) > abs(yd)

                        if yr > 0
                            classifications[i] = COUNTERCLOCKWISE_ROTATION
                        else
                            classifications[i] = CLOCKWISE_ROTATION
                        end
                    else
                        
                        if yd > 0
                            classifications[i] = POSITIVE_SCALING
                        else
                            classifications[i] = NEGATIVE_SCALING
                        end
                    end
                end

                if classifications[1] == classifications[2] == classifications[3]
                    #p = plot!(Shape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]]), fill=eigenvalueColors[classifications[1]], linecolor=eigenvalueColors[classifications[1]], legend=false)
                    p = plotShape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]], eigenvalueColors[classifications[1]], show_borders)

                    classification = classifications[1]
                    # Plot elliptical regions that can only occur inside of a cell
                    if classification != ANISOTROPIC_STRETCHING

                        t1, t2, t3 = tensors

                        M = [ positions[1][2] positions[2][2] positions[3][2] ;
                        positions[1][3] positions[2][3] positions[3][3] ;
                        1               1               1               ]
                  
                        if classification == POSITIVE_SCALING || classification == NEGATIVE_SCALING
                            # we look for where yd = ys
                            K = [ t1[1,1]-t1[2,2] t2[1,1]-t2[2,2] t3[1,1]-t3[2,2] ;
                                  t1[2,1]+t1[1,2] t2[2,1]+t2[1,2] t3[2,1]+t2[1,2] ;
                                  t1[1,1]+t1[2,2] t2[1,1]+t2[2,2] t3[1,1]+t3[2,2] ]
                        else
                            # we look for where yd = yr
                            # we look for where yd = ys
                            K = [ t1[1,1]-t1[2,2] t2[1,1]-t2[2,2] t3[1,1]-t3[2,2] ;
                                  t1[2,1]+t1[1,2] t2[2,1]+t2[1,2] t3[2,1]+t2[1,2] ;
                                  t1[2,1]-t1[1,2] t2[2,1]-t2[1,2] t3[2,1]-t3[1,2] ]
                        end

                        # Derive ellipse coefficients
                        M_inverse = M^-1

                        C = K*M_inverse

                        A = C[1,1]^2 + C[2,1]^2 - C[3,1]^2
                        B = 2*(C[1,1]*C[1,2] + C[2,1]*C[2,2] - C[3,1]*C[3,2])
                        C_ = C[1,2]^2 + C[2,2]^2 - C[3,2]^2
                        D = 2*(C[1,1]*C[1,3] + C[2,1]*C[2,3] - C[3,1]*C[3,3])
                        E = 2*(C[1,2]*C[1,3] + C[2,2]*C[2,3] - C[3,2]*C[3,3])
                        F = C[1,3]^2 + C[2,3]^2 - C[3,3]^2

                        if A < 0
                            A *= -1
                            B *= -1
                            C *= -1
                            D *= -1
                            E *= -1
                            F *= -1
                        end

                        discriminant = B^2 - 4*A*C_

                        # Check if the conic is indeed an ellipse. If not, we don't have an ellipse inside of the cell obviously.
                        if discriminant < 0
    
                            # Check if a vertex is contained within the ellipse
                            _, vx, vy = positions[1]
                            ellipse_value = A*vx^2 + B*vx*vy + C_*vy^2 + D*vx + E*vy + F
    
                            # In this case, the vertex is not within the ellipse meaning that the ellipse is either entirely
                            # inside or outside the cell
                            if ellipse_value > 0
                                # Check to see if the center is inside of the cell.
                                center = [ 2*C_*D-B*E ; 2*A*E-B*D ; discriminant ] / discriminant
                                barrycenter_vals = M_inverse*center
                                if 0 <= barrycenter_vals[1] <= 1 && 0 <= barrycenter_vals[2] <= 1 && 0 <= barrycenter_vals[3] <= 1
                                    # If the center is in the cell, then plot the ellipse as a dot.
                                    center_tensor = barrycenter_vals[1]*t1 + barrycenter_vals[2]*t2 + barrycenter_vals[3]*t3
                                    center_classification = classifyTensorEigenvalue(center_tensor)
                                    push!(scatterPoints, (center[1], center[2], eigenvectorColors[center_classification]))
                                end
                            end
                        end


                    end

                else
                    # Plot the color dividing that is not related to anisotropic stretching

                    # each thing here has 2 variables, one for the case where we are checking if yd = yr and another for yd = -yr

                    # each stores edgee #, t, edge #, t, where t is used to interpolate

                    equality_edges = [-1 -1 ; -1  -1 ]
                    interp_values = [ -1.0 -1.0 ; -1.0 -1.0 ]
                    edges_used = [ false false false ; false false false ]
                    signs = [1,-1]

                    # stores the vertex that is enclosed within the triangle from each line.
                    # Equal to -1 if no edge is covered
                    vertices_covered = [-1, -1]
                    matchSign = 0

                    sign_(x) = sign(x)

                    # Fill out the lists above via linear interpolation
                    for sign in 1:2

                        for edge in 1:3

                            dif1 = decompositions[edges[edge, 1]][1] - signs[sign]*decompositions[edges[edge, 1]][2]
                            dif2 = decompositions[edges[edge, 2]][1] - signs[sign]*decompositions[edges[edge, 2]][2]

                            if sign_(dif1) != sign_(dif2)
                                interp = dif2/(dif2-dif1)
                                if equality_edges[sign, 1] == -1
                                    equality_edges[sign, 1] = edge
                                    interp_values[sign, 1] = interp
                                else
                                    equality_edges[sign, 2] = edge
                                    interp_values[sign, 2] = interp
                                end

                                edges_used[sign, edge] = true
                                matchSign = signs[sign]

                            end

                        end

                        if equality_edges[sign, 1] != -1
                            for edge in 1:3
                                if !(edges_used[sign, edge])
                                    vertices_covered[sign] = edge
                                end
                            end
                        end

                    end

                    # remaining_vertex is a/the vertex that is not covered by any of these lines. It will be the base color for the triangle.

                    remaining_vertices = [true, true, true]
                    if vertices_covered[1] != -1
                        remaining_vertices[vertices_covered[1]] = false
                    end
                    if vertices_covered[2] != -1
                        remaining_vertices[vertices_covered[2]] = false
                    end

                    remaining_vertex = -1
                    for v in 1:3
                        # If white does not need to be used as the base color then we do not set it as the base color
                        if (remaining_vertex == -1 && remaining_vertices[v]) || (remaining_vertices[v] && classifications[v] != ANISOTROPIC_STRETCHING)
                            remaining_vertex = v
                        end
                    end

                    # Do the actual drawing

                    if classifications[remaining_vertex] != ANISOTROPIC_STRETCHING || (vertices_covered[1] != -1 && vertices_covered[2] != -1)
                        fillColor = eigenvalueColors[classifications[remaining_vertex]]
                    else
                        if vertices_covered[1] != -1
                            fillColor = eigenvalueColors[eigenvalueRegionBorders[1][classifications[vertices_covered[1]]]]
                        else
                            fillColor = eigenvalueColors[eigenvalueRegionBorders[-1][classifications[vertices_covered[2]]]]
                        end
                    end

                    #p = plot!(Shape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]]), fill=eigenvalueColors[classifications[remaining_vertex]], legend=false)
                    p = plotShape([positions[1][2], positions[2][2], positions[3][2]], [positions[1][3], positions[2][3], positions[3][3]], fillColor, show_borders)

                    for sign in 1:2

                        # Draw other triangles as needed
                        if equality_edges[sign,1] != -1

                            # x0, x1, y0, y1
                            endpoint_coords = [-1.0, -1.0, -1.0, -1.0]
                            for endpoint in 1:2
                                _, v1x, v1y = positions[edges[equality_edges[sign,endpoint],1]]
                                _, v2x, v2y = positions[edges[equality_edges[sign,endpoint],2]]
                                interp = interp_values[sign,endpoint]

                                endpoint_coords[endpoint] = interp*v1x + (1-interp)*v2x
                                endpoint_coords[2+endpoint] = interp*v1y + (1-interp)*v2y
                            end

                            #p = plot!(Shape([endpoint_coords[1], endpoint_coords[2], positions[vertices_covered[sign]][2]], [endpoint_coords[3], endpoint_coords[4], positions[vertices_covered[sign]][3]]), fill=eigenvalueColors[classifications[vertices_covered[sign]]], legend=false)
                            if classifications[vertices_covered[sign]] != ANISOTROPIC_STRETCHING
                                p = plotShape([endpoint_coords[1], endpoint_coords[2], positions[vertices_covered[sign]][2]], [endpoint_coords[3], endpoint_coords[4], positions[vertices_covered[sign]][3]], eigenvalueColors[classifications[vertices_covered[sign]]], show_borders)
                            elseif classifications[remaining_vertex] != ANISOTROPIC_STRETCHING
                                p = plotShape([endpoint_coords[1], endpoint_coords[2], positions[vertices_covered[sign]][2]], [endpoint_coords[3], endpoint_coords[4], positions[vertices_covered[sign]][3]], eigenvalueColors[eigenvalueRegionBorders[signs[sign]][classifications[remaining_vertex]]], show_borders)
                            end
                            push!(borderLines1, ( endpoint_coords[1], endpoint_coords[2], endpoint_coords[3], endpoint_coords[4] ))
                        end

                    end

                    # Fill in the regions corresponding to anisotropic stretching.

                    # Loop through the edges identifying points and add them to the list in circular order.
                    # That is, for each edge (v1,v2), in order as you travel from v1 -> v2. That is, large interp values to small.
                    # Only process edges that have different types.
                    # If there is an edge with no anisotropic stretching on it, and also 
                    # look for a junction point and put that in place, if this has not already occurred

                    tried_junction = false
                    points_x::Array{Float64} = []
                    points_y::Array{Float64} = []
                    points_edges::Array{Int64} = []
                    modes = [0, 0, 0]

                    for edge in 1:3
                        if classifications[edges[edge,1]] != classifications[edges[edge,2]]

                            interp_values = []
                            t1 = tensors[edges[edge,1]]
                            t2 = tensors[edges[edge,2]]
                            found_term = false                            

                            K = [t1[1,1]+t1[2,2] t1[2,1]-t1[1,2] t1[1,1]-t1[2,2] t1[1,2]+t1[2,1] ; 
                                 t2[1,1]+t2[2,2] t2[2,1]-t2[1,2] t2[1,1]-t2[2,2] t2[1,2]+t2[2,1] ]
                            
                            α_0s = K[2,3]^2 + K[2,4]^2                                 
                            α_1s = 2*K[2,3]*(K[1,3] - K[2,3]) + 2*K[2,4]*(K[1,4] - K[2,4])
                            α_2s = (K[1,3] - K[2,3])^2 + (K[1,4] - K[2,4])^2

                            for line in 1:2

                                α_0 = α_0s - K[2,line]^2
                                α_1 = α_1s - 2*K[2,line]*(K[1,line] - K[2,line])
                                α_2 = α_2s - (K[1,line] - K[2,line])^2

                                discriminant = α_1^2 - 4*α_0*α_2
                                if discriminant >= 0
                                    candidate_1 = (-α_1 + sqrt(discriminant)) / (2*α_2)
                                    candidate_2 = (-α_1 - sqrt(discriminant)) / (2*α_2)

                                    candidates = [max(candidate_1,candidate_2), min(candidate_1,candidate_2)]
                                    for c in candidates

                                        if 0 <= c <= 1

                                            interp_tensor = c*t1 + (1-c)*t2
                                            yd, yr, ys, _ = decomposeTensor(interp_tensor)
                                            if ys^2 >= yd^2-ϵ && ys^2 >= yr^2-ϵ
                                                push!(interp_values, c)
                                                found_term = true
                                                modes[1] += 1
                                            end
                                        end
                                    end
                                end
                            end

                            if found_term

                                sort!(interp_values, rev=true)

                                _, v1x, v1y = positions[edges[edge,1]]
                                _, v2x, v2y = positions[edges[edge,2]]

                                for c in interp_values
                                    push!(points_x, c*v1x + (1-c)*v2x)
                                    push!(points_y, c*v1y + (1-c)*v2y)
                                    push!(points_edges, edge)
                                end

                            elseif !tried_junction

                                tried_junction = true
                                
                                t1, t2, t3 = tensors

                                # Look for the junction:

                                K = [ t1[1,1]+t1[2,2] t2[1,1]+t2[2,2] t3[1,1]+t3[2,2] ;
                                      t1[2,1]-t1[1,2] t2[2,1]-t2[1,2] t3[2,1]-t3[1,2] ;
                                      t1[1,1]-t1[2,2] t2[1,1]-t2[2,2] t3[1,1]-t3[2,2] ;
                                      t1[2,1]+t1[1,2] t2[2,1]+t2[1,2] t3[2,1]+t3[1,2] ]

                                M = [ positions[1][2] positions[2][2] positions[3][2] ;
                                      positions[1][3] positions[2][3] positions[3][3] ;
                                      1               1               1               ]
                                
                                M_inverse = M^-1

                                C = K*M_inverse

                                A = C[3,1]^2 + C[4,1]^2 - C[2,1]^2
                                B = 2*(C[3,1]*C[3,2] + C[4,1]*C[4,2] - C[2,1]*C[2,2])
                                C_ = C[3,2]^2 + C[4,2]^2 - C[2,2]^2
                                D = 2*(C[3,1]*C[3,3] + C[4,1]*C[4,3] - C[2,1]*C[2,3])
                                E = 2*(C[3,2]*C[3,3] + C[4,2]*C[4,3] - C[2,2]*C[2,3])
                                F = C[3,3]^2 + C[4,3]^2 - C[2,3]^2
                                U = C[1,1] - C[2,1]
                                V = C[1,2] - C[2,2]
                                W = C[1,3] - C[2,3]

                                α_0 = C_*W^2/(V^2) - E*W/V + F
                                α_1 = D - B*W/V + 2*C_*U*W/(V^2) - E*U/V
                                α_2 = A - B*U/V + C_*(U^2)/(V^2)

                                discriminant = α_1^2 - 4*α_0*α_2
                                if discriminant >= 0
                                    x_candidate_1 = (-α_1 + sqrt(discriminant))/(2*α_2)
                                    x_candidate_2 = (-α_1 - sqrt(discriminant))/(2*α_2)

                                    y_candidate_1 = -U*x_candidate_1/V - W/V
                                    y_candidate_2 = -U*x_candidate_2/V - W/V

                                    # bc = barrycenter cooordinates
                                    bc = M_inverse * [ x_candidate_1 x_candidate_2 ; y_candidate_1 y_candidate_2 ; 1 1 ]

                                    if 0 <= bc[1,1] <= 1 && 0 <= bc[2,1] <= 1 && 0 <= bc[3,1] <= 1
                                        push!(points_x, x_candidate_1)
                                        push!(points_y, y_candidate_1)
                                        push!(points_edges, -1)

                                        modes[2] += 1
                                    elseif 0 <= bc[1,2] <= 1 && 0 <= bc[2,2] <= 1 && 0 <= bc[3,2] <= 1
                                        push!(points_x, x_candidate_2)
                                        push!(points_y, y_candidate_2)
                                        push!(points_edges, -1)

                                        modes[2] += 1
                                    end

                                end

                            end
                        end

                        if classifications[edges[edge,2]] == ANISOTROPIC_STRETCHING
                            push!(points_x, positions[edges[edge,2]][2] )
                            push!(points_y, positions[edges[edge,2]][3] )
                            push!(points_edges, -2)
                            modes[3] += 1
                        end

                    end

                    if length(points_x)  > 0
                        #p = plot!( Shape(points_x, points_y), color="white", fill=eigenvalueColors[ANISOTROPIC_STRETCHING], legend=false )
                        #plotShape(points_x, points_y, eigenvalueColors[ANISOTROPIC_STRETCHING], show_borders)
                        push!(anisotropicShapes, Shape(points_x, points_y))

                        number = length(points_x)
                        push!(points_edges, points_edges[1])
                        push!(points_x, points_x[1])
                        push!(points_y, points_y[1])

                        for segment in 1:number
                            if points_edges[segment] != points_edges[segment+1] && points_edges[segment] != -2 && points_edges[segment+1] != -2
                                push!(borderLines2, (points_x[segment], points_x[segment+1], points_y[segment], points_y[segment+1] ))
                            end
                        end

                    end
                end
            end
        end
    end

    for l in borderLines1
        p = plot!( [l[1], l[2]], [l[3], l[4]], color="black", legend=false, width=1.5 )
    end

    anisotropicColor = eigenvalueColors[ANISOTROPIC_STRETCHING]
    if show_borders
        for s in anisotropicShapes
            p = plot!( s, fill=anisotropicColor, legend=false )
        end
    else
        for s in anisotropicShapes
            p = plot!( s, fill=anisotropicColor, linecolor=anisotropicColor, linewidth=1.5, legend=false )
        end
    end

    if !show_borders
        for l in borderLines2
            p = plot!( [l[1], l[2]], [l[3], l[4]], color="black", legend=false, width=1.5 )
        end
    end

    if show_points
        for i in xMin:(xMax+1)
            for j in yMin:(yMax+1)
                tensor = getTensor(tf, t, i, j)
                classification = classifyTensorEigenvalue(tensor)
                p = scatter!([i], [j], color=eigenvalueColors[classification], markersize=2 )
            end
        end
    end

    return p

end

function plotDegenerateLines(tf::TensorField2d, t::Int64, color=["black", "black"])
    x, y = tf.dims[2], tf.dims[3]
    x -= 1
    y -= 1

    p = plot([], [], legend=false, color="black")

    for i in 1:x
        for j in 1:y
            for k in 0:1
                positions = getCellVertexCoords(t, i, j, Bool(k))
                tensors = getTensorsAtCell(tf, t, i, j, Bool(k))

                _, yr1, ys1, θ_1 = decomposeTensor(tensors[1])
                _, yr2, ys2, θ_2 = decomposeTensor(tensors[2])
                _, yr3, ys3, θ_3 = decomposeTensor(tensors[3])

                positive_difs = [yr1 - ys1, yr2 - ys2, yr3 - ys3]
                negative_difs = [yr1 + ys1, yr2 + ys2, yr3 + ys3]

                combinations = [(1,2), (1,3), (2,3)]

                positive_intercepts = []
                negative_intercepts = []

                for c in combinations
                    pos1 = positive_difs[c[1]]
                    pos2 = positive_difs[c[2]]

                    neg1 = negative_difs[c[1]]
                    neg2 = negative_difs[c[2]]

                    if pos1*pos2 < 0
                        t_ = pos2/(pos2-pos1)
                        x_ = t_ * positions[c[1]][2] + (1-t_) * positions[c[2]][2]
                        y_ = t_ * positions[c[1]][3] + (1-t_) * positions[c[2]][3]
                        push!(positive_intercepts, (x_, y_))
                    end

                    if neg1*neg2 < 0
                        t_ = neg2/(neg2-neg1)
                        x_ = t_ * positions[c[1]][2] + (1-t_) * positions[c[2]][2]
                        y_ = t_ * positions[c[1]][3] + (1-t_) * positions[c[2]][3]
                        push!(negative_intercepts, (x_, y_))
                    end
                end

                if length(positive_intercepts) == 2
                    p = plot!( [ positive_intercepts[1][1], positive_intercepts[2][1] ], [ positive_intercepts[1][2], positive_intercepts[2][2] ], legend=false, color=color[1] )
                end

                if length(negative_intercepts) == 2
                    p = plot!( [ negative_intercepts[1][1], negative_intercepts[2][1] ], [ negative_intercepts[1][2], negative_intercepts[2][2] ], legend=false, color=color[2] )
                end

            end
        end
    end

    return p
    
end

end