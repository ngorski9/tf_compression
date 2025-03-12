using WriteVTK
using LinearAlgebra

function main()
    println("hi")

    folder = "../output/slice"
    saveName = "../test.png"
    dims = (101,101)
    scale = 10 # how many extra points do we add between the actual grid points (so the quadratic interp is more accurate).
    val = true




    dp = ( 224, 142, 69 )
    rp = ( 155, 39, 51 )
    dn = ( 201, 168, 245 )
    rn = ( 49, 59, 142 )
    s = ( 234, 234, 234 )

    rrp = ( 155, 39, 51 )
    srp = ( 193, 153, 116 )
    sym = ( 173, 219, 240 )
    srn = (133, 153, 188 )
    rrn = (46, 59, 142 )

    a_x = scale*(dims[1]-1)+dims[1]
    a_y = scale*(dims[2]-1)+dims[2]

    # experimental...
    img = Array{UInt8}(undef, (3,a_x,a_y))
    frobenius = Array{Float64}(undef, (a_x, a_y))
    categorical = Array{Float64}(undef, (a_x, a_y))

    # define set functions
    function set_color(x,y,c)
        img[]
    end


    a_file = open("$folder/row_1_col_1.dat", "r")
    b_file = open("$folder/row_1_col_2.dat", "r")
    c_file = open("$folder/row_2_col_1.dat", "r")
    d_file = open("$folder/row_2_col_2.dat", "r")

    a_array = reshape( reinterpret( Float64, read(a_file) ), dims )
    b_array = reshape( reinterpret( Float64, read(b_file) ), dims )
    c_array = reshape( reinterpret( Float64, read(c_file) ), dims )
    d_array = reshape( reinterpret( Float64, read(d_file) ), dims )

    close(a_file)
    close(b_file)
    close(c_file)
    close(d_file)

    cp_vertices = zeros(Bool, dims)
    cp_locs::Array{Tuple{Float64,Float64}} = Array{Tuple{Float64,Float64}}(undef, 0)

    for j in 1:dims[2]-1
        for i in 1:dims[1]-1

            a1 = a_array[i,j]
            a2 = a_array[i+1,j]
            a3 = a_array[i,j+1]
            a4 = a_array[i+1,j+1]

            b1 = b_array[i,j]
            b2 = b_array[i+1,j]
            b3 = b_array[i,j+1]
            b4 = b_array[i+1,j+1]

            c1 = c_array[i,j]
            c2 = c_array[i+1,j]
            c3 = c_array[i,j+1]
            c4 = c_array[i+1,j+1]

            d1 = d_array[i,j]
            d2 = d_array[i+1,j]
            d3 = d_array[i,j+1]
            d4 = d_array[i+1,j+1]

            if !val
                # extract degenerate points.
                Δ1 = (a1-d1)/2
                Δ2 = (a2-d2)/2
                Δ3 = (a3-d3)/2
                Δ4 = (a4-d4)/2

                F1 = (b1+c1)/2
                F2 = (b2+c2)/2
                F3 = (b3+c3)/2
                F4 = (b4+c4)/2

                # Mat1 = [Δ1 Δ2 Δ3 ; F1 F2 F3 ; 1 1 1]
                # if det(Mat1) != 0
                #     sol1 = (Mat1^-1) * [0 ; 0 ; 1]
                #     if 0 <= sol1[1] <= 1 && 0 <= sol1[2] <= 1 && 0 <= sol1[3] <= 1
                #         if sol1[1] == 1.0
                #             if !cp_vertices[(i-1)*scale,(j-1)*scale]
                #                 push!(cp_locs, (i,j))
                #                 cp_vertices[i,j] = true
                #             end
                #         push!(cp_locs, )
                #     end                    println((px_y,px_x))
                # end

                # Mat2 = [Δ2 Δ3 Δ4 ; F2 F3 F4 ; 1 1 1]
                # sol2 = (Mat2^-1) * [0 ; 0 ; 1]
            end

            for cell_y in 0:1+scale
                for cell_x in 0:1+scale

                    if cell_y == 1+scale && j != dims[2]-1
                        continue
                    end

                    if cell_x == 1+scale && i != dims[1]-1
                        continue
                    end


                    if cell_x + cell_y <= (scale + 1.0)
                        t2 = cell_x / (scale+1.0)
                        t3 = cell_y / (scale+1.0)
                        t4 = 0.0

                        t1 = 1.0-t2-t3
                    else
                        t1 = 0.0
                        t2 = 1.0 - cell_y / (scale + 1.0)
                        t3 = 1.0 - cell_x / (scale + 1.0)
                        t4 = 1.0 - t2 - t3
                    end

                    a = t1*a1 + t2*a2 + t3*a3 + t4*a4
                    b = t1*b1 + t2*b2 + t3*b3 + t4*b4
                    c = t1*c1 + t2*c2 + t3*c3 + t4*c4
                    d = t1*d1 + t2*d2 + t3*d3 + t4*d4

                    D = (a+d)/2
                    R = (c-b)/2
                    S = sqrt( (a-d)^2 + (b+c)^2 )/2

                    px_x = (i-1)*(scale+1) + cell_x + 1
                    px_y = (j-1)*(scale+1) + cell_y + 1

                    frobenius[px_x,a_y-px_y+1] = sqrt(a^2+b^2+c^2+d^2)

                    if val
                        if abs(D) >= abs(R) && abs(D) >= abs(S)
                            if D > 0
                                img[:,px_x,a_y-px_y+1] .= dp
                                categorical[px_x,a_y-px_y+1] = 5.0
                            else
                                img[:,px_x,a_y-px_y+1] .= dn
                                categorical[px_x,a_y-px_y+1] = 4.0
                            end
                        elseif abs(R) >= abs(D) && abs(R) >= abs(S)
                            if R > 0
                                img[:,px_x,a_y-px_y+1] .= rp
                                categorical[px_x,a_y-px_y+1] = 3.0
                            else
                                img[:,px_x,a_y-px_y+1] .= rn
                                categorical[px_x,a_y-px_y+1] = 2.0
                            end
                        else
                            img[:,px_x,a_y-px_y+1] .= s
                            categorical[px_x,a_y-px_y+1] = 1.0
                        end
                    else
                        if R == 0.0
                            img[:,px_x,a_y-px_y+1] .= sym
                            categorical[px_x,a_y-px_y+1] = 1.0
                        elseif R > 0
                            if S > abs(R)
                                img[:,px_x,a_y-px_y+1] .= srp
                                categorical[px_x,a_y-px_y+1] = 2.0
                            else
                                img[:,px_x,a_y-px_y+1] .= rrp
                                categorical[px_x,a_y-px_y+1] = 3.0
                            end
                        else
                            if S > abs(R)
                                img[:,px_x,a_y-px_y+1]  .= srn
                                categorical[px_x,a_y-px_y+1] = 4.0
                            else
                                img[:,px_x,a_y-px_y+1] .= rrn
                                categorical[px_x,a_y-px_y+1] = 5.0
                            end
                        end
                    end # end if val or vec

                end # end for cell_x
            end # end for cell_y
        end # end for i
    end # end for j

    vtk_grid("../elevation", 0:1:a_y-1,0:1:a_x-1,0:1:0) do vtk
        vtk["frobenius"] = frobenius
        vtk["categorical"] = categorical
        vtk["image"] = img
    end

    # cells::Vector{MeshCell} = []

    # randPoints = rand(3,10)*a_x
    # pointsColor = rand(10)
    # vtk_grid("../test",randPoints,cells) do vtk
    #     vtk["color"] = pointsColor
    # end
    
end

main()