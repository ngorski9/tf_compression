using LinearAlgebra
using DataStructures
using Random
using PyCall

function extractCP(a_array, b_array, c_array, d_array, size, trisector_points, wedge_points)
    # identify critical points
    for j in 1:size[2]-1
        for i in 1:size[1]-1
            for k in 0:1
                
                if k == 0
                    # bottom
                    x1, y1 = (i,j)
                    x2, y2 = (i+1,j)
                    x3, y3 = (i,j+1)
                else
                    # top
                    x1, y1 = (i,j+1)
                    x2, y2 = (i+1,j)
                    x3, y3 = (i+1,j+1)
                end

                l12 = (a_array[x1,y1] - d_array[x1,y1])*(b_array[x2,y2]+c_array[x2,y2]) - (a_array[x2,y2] - d_array[x2,y2])*(b_array[x1,y1]+c_array[x1,y1])
                l23 = (a_array[x2,y2] - d_array[x2,y2])*(b_array[x3,y3]+c_array[x3,y3]) - (a_array[x3,y3] - d_array[x3,y3])*(b_array[x2,y2]+c_array[x2,y2])
                l31 = (a_array[x3,y3] - d_array[x3,y3])*(b_array[x1,y1]+c_array[x1,y1]) - (a_array[x1,y1] - d_array[x1,y1])*(b_array[x3,y3]+c_array[x3,y3])

                cp = 0

                if l12 < 0 && l23 < 0 && l31 < 0
                    # wedge
                    cp = 1
                elseif l12 > 0 && l23 > 0 && l31 > 0
                    # trisector
                    cp = 2
                end

                if cp == 1 || cp == 2
                    mat = [ (a_array[x1,y1] - d_array[x1,y1]) (a_array[x2,y2] - d_array[x2,y2]) (a_array[x3,y3] - d_array[x3,y3]) ; b_array[x1,y1]+c_array[x1,y1] b_array[x2,y2]+c_array[x2,y2] b_array[x3,y3]+c_array[x3,y3] ; 1 1 1 ]
                    μ = (mat^-1) * [0 ; 0 ; 1]

                    cx = μ[1] * Float64(x1) + μ[2] * Float64(x2) + μ[3] * Float64(x3)
                    cy = μ[1] * Float64(y1) + μ[2] * Float64(y2) + μ[3] * Float64(y3)

                    if cp == 1
                        push!(wedge_points, (cx, cy))
                    else
                        push!(trisector_points, (cx,cy))
                    end

                end

            end
        end
    end
end

function randomNoise(array)
    width, height = size(array)
    for j in 1:height
        for i in 1:width
            array[i,j] = rand()
        end
    end
end

function randomCircles(array, radius)
    width, height = size(array)

    for i in 1:width*height/radius
        centerX = Int64(round(rand() * (width-1))+1)
        centerY = Int64(round(rand() * (height-1))+1)
        intensity = rand()
        for xo in -radius:radius
            for yo in -radius:radius
                if xo^2 + yo^2 <= radius && 1 <= centerX + xo <= width && 1 <= centerY + yo <= height
                    array[centerX+xo, centerY+yo] = intensity
                end
            end
        end

    end

end

struct TensorField
    size::Tuple{Int64,Int64}
    a::Matrix{Float64}
    b::Matrix{Float64}
    c::Matrix{Float64}
    d::Matrix{Float64}
end

function inBounds(tf::TensorField, x::Vector{Float64})
    return x[1] >= 1 && x[1] <= tf.size[1] && x[2] >= 1 && x[2] <= tf.size[2]
end

function dot(a::Vector{Float64}, b::Vector{Float64})
    return a[1]*b[1] + a[2]*b[2]
end

function vector(A::Array{Float64}, dual=false, scale::Float64=0.0)

    if A[1,1] == 0.0 && A[1,2] == 0.0 && A[2,1] == 0.0 && A[2,2] == 0.0
        return [0.0,0.0]
    end

    if dual

        d = (A[1,1] + A[2,2]) / 2
        r = (A[2,1] - A[1,2]) / 2

        deviator = (A - d*[1 0 ; 0 1] - r*[0 -1 ; 1 0])/r
        cplx = deviator[1,1] + deviator[1,2]*im
        θ = angle(cplx) + pi/2

        if r > 0
            vector = [ sin(θ), 1 - cos(θ) ]
        else
            vector = [ cos(θ) - 1, sin(θ) ]
        end

        vector /= norm(vector)

        return vector

    else
        vals = eigvals(A)
        if vals[1] > vals[2]
            if scale == 0
                return eigvecs(A)[:,1]
            else
                return scale*abs(vals[1])*eigvecs(A)[:,1]
            end
        else
            if scale == 0
                return eigvecs(A)[:,2]
            else
                return scale*abs(vals[2])*eigvecs(A)[:,2]
            end
        end
    end

end

function interpolate(tf::TensorField,v::Vector{Float64})
    x,y = v
    xFloor = Int64(floor(x))
    yFloor = Int64(floor(y))
    xFrac = x - floor(x)
    yFrac = y - floor(y)
    if xFrac - yFrac <= 1
        # bottom cell
        a = tf.a[xFloor + 1, yFloor] * xFrac + tf.a[xFloor, yFloor + 1] * yFrac + tf.a[xFloor, yFloor] * (1 - xFrac - yFrac)
        b = tf.b[xFloor + 1, yFloor] * xFrac + tf.b[xFloor, yFloor + 1] * yFrac + tf.b[xFloor, yFloor] * (1 - xFrac - yFrac)
        c = tf.c[xFloor + 1, yFloor] * xFrac + tf.c[xFloor, yFloor + 1] * yFrac + tf.c[xFloor, yFloor] * (1 - xFrac - yFrac)
        d = tf.d[xFloor + 1, yFloor] * xFrac + tf.d[xFloor, yFloor + 1] * yFrac + tf.d[xFloor, yFloor] * (1 - xFrac - yFrac)

        return [a b ; c d]
    else
        # top cell
        a = tf.a[xFloor, yFloor + 1] * (1 - xFrac) + tf.a[xFloor + 1, yFloor] * (1 - yFrac) + tf.a[xFloor + 1, yFloor + 1] * (xFrac + yFrac - 1)
        a = tf.b[xFloor, yFloor + 1] * (1 - xFrac) + tf.b[xFloor + 1, yFloor] * (1 - yFrac) + tf.b[xFloor + 1, yFloor + 1] * (xFrac + yFrac - 1)
        a = tf.c[xFloor, yFloor + 1] * (1 - xFrac) + tf.c[xFloor + 1, yFloor] * (1 - yFrac) + tf.c[xFloor + 1, yFloor + 1] * (xFrac + yFrac - 1)
        a = tf.d[xFloor, yFloor + 1] * (1 - xFrac) + tf.d[xFloor + 1, yFloor] * (1 - yFrac) + tf.d[xFloor + 1, yFloor + 1] * (xFrac + yFrac - 1)

        return [a b ; c d]
    end

end

# expensive lol
function rk4_vector(tf::TensorField, x::Vector{Float64}, δ::Float64, lastVector::Vector{Float64}, dual::Bool, evecScale::Float64=0.0)

    t1 = interpolate(tf, x)
    e1 = vector(t1, dual, evecScale)
    if dot(lastVector, e1) < 0
        e1 *= -1
    end
    a = δ * e1
    xa = x + 0.5a
    if !inBounds(tf, xa)
        return [Inf,Inf]
    end

    t2 = interpolate(tf, xa)
    e2 = vector(t2, dual, evecScale)
    if dot(lastVector, e2) < 0
        e2 *= -1
    end
    b = δ * e2
    xb = x + 0.5b
    if !inBounds(tf, xb)
        return [Inf,Inf]
    end

    t3 = interpolate(tf, xb)
    e3 = vector(t3, dual, evecScale)
    if dot(lastVector, e3) < 0
        e3 *= -1
    end
    c = δ * e3
    xc = x + c # yes this is correct. This should not be x + 0.5c
    if !inBounds(tf, xc)
        return [Inf,Inf]
    end

    t4 = interpolate(tf, xc)
    e4 = vector(t4, dual, evecScale)
    if dot(lastVector, e4) < 0
        e4 *= -1
    end
    d = δ * e4

    if dot(lastVector, a+2b+2c+d) < 0
        println("something is wrong")
        println((e1,e2,e3,e4))
        println((a,b,c,d))
        println(lastVector)
        println((dot(lastVector, a), dot(lastVector,b), dot(lastVector,c), dot(lastVector,d)))
    end

    return (a + 2b + 2c + d) / 6

end

function to_pixel(x, scale)
    return (Int64(floor( (x[1]-1)*scale+1 )), Int64(floor( (x[2]-1)*scale+1 )))
end

function pixel_near(px,list)
    for j in -2:2
        for i in -2:2
            if (px[1]+i, px[2]+j) in list
                return true
            end
        end
    end
    return false
end

function main()
    Random.seed!(1)
    t1 = time()
    plt = pyimport("matplotlib.pyplot")
    colors = pyimport("matplotlib.colors")

    folder = "../output/reconstructed"
    size = (65, 65)
    scale = 2
    evecScale=0.0
    power = 2.5

    alt_folder = ""
    load_lic = ""
    save_lic = ""

    # old settings: num_steps: 60
    # max_path_tracing = 60
    # block_size = 20
    # delta: 0.1

    num_steps = 180 # Interpolation length
    max_path_tracing = 180 # kill after a certain number of steps to avoid loops
    block_size = 20
    hit_threshold = 8
    δ = 0.01
    # δ = 0.1
    asymmetric = false

    # load and set up the tensor field
    a = time()
    a_file = open("$folder/row_1_col_1.dat", "r")
    b_file = open("$folder/row_1_col_2.dat", "r")
    d_file = open("$folder/row_2_col_2.dat", "r")

    a_array = reshape( reinterpret( Float64, read(a_file) ), size )
    b_array = reshape( reinterpret( Float64, read(b_file) ), size )
    d_array = reshape( reinterpret( Float64, read(d_file) ), size )

    if asymmetric
        c_file = open("$folder/row_2_col_1.dat", "r")
        c_array = reshape( reinterpret( Float64, read(c_file) ), size )
    else
        c_array = b_array
    end

    tf = TensorField(size, a_array, b_array, c_array, d_array)

    wedge_points = []
    trisector_points = []

    wedge_points_alt = []
    trisector_points_alt = []

    # extract critical points
    extractCP(a_array,b_array,c_array,d_array,size,trisector_points,wedge_points)

    if alt_folder != ""
        a_file_alt = open("$alt_folder/row_1_col_1.dat", "r")
        b_file_alt = open("$alt_folder/row_1_col_2.dat", "r")
        d_file_alt = open("$alt_folder/row_2_col_2.dat", "r")
    
        a_array_alt = reshape( reinterpret( Float64, read(a_file_alt) ), size )
        b_array_alt = reshape( reinterpret( Float64, read(b_file_alt) ), size )
        d_array_alt = reshape( reinterpret( Float64, read(d_file_alt) ), size )
    
        if asymmetric
            c_file_alt = open("$alt_folder/row_2_col_1.dat", "r")
            c_array_alt = reshape( reinterpret( Float64, read(c_file_alt) ), size )
        else
            c_array_alt = b_array_alt
        end

        extractCP(a_array_alt,b_array_alt,c_array_alt,d_array_alt,size,trisector_points_alt,wedge_points_alt)
    end

    # set up the input texture and other images

    imageSize = ( (size[1]-1)*scale, (size[2]-1)*scale )
    noise = zeros(Float64, imageSize )
    # randomNoise(noise)
    randomCircles(noise, scale-1)
    for j in 1:imageSize[2]
        for i in 2:imageSize[1]
            xo = (i-0.5)/scale+1.0
            yL = (j-0.7)/scale+1.0
            yH = (j-0.3)/scale+1.0

            if interpolate(tf, [xo, yL] ) == [0.0 0.0 ; 0.0 0.0] && interpolate(tf, [xo,yH]) == [0.0 0.0 ; 0.0 0.0]
                noise[i,j] = 0.0
            end

        end
    end

    finalImage = zeros(Float64, imageSize)

    if load_lic != ""
        inf = open(load_lic, "r")
        inBytes = reshape(reinterpret(Float64,read(inf)),imageSize)
        for j in 1:imageSize[2]
            for i in 1:imageSize[1]
                finalImage[i,j] = inBytes[i,j]
            end
        end
    else
        # finalImage = noise
        numHits = zeros(Int64, imageSize)

        # iterate and create streamlines
        blocks_dims = ( Int64(ceil(imageSize[1] / block_size)), Int64(ceil(imageSize[2] / block_size)) )

        numSkips = 0

        for y in 1:block_size
            for x in 1:block_size
                println((x,y,numSkips))
                for by = 1:blocks_dims[2]
                    for bx in 1:blocks_dims[1]

                        px = (bx-1) * block_size + x
                        py = (by-1) * block_size + y

                        if px <= imageSize[1] && py <= imageSize[2]

                            if numHits[px,py] >= hit_threshold
                                numSkips += 1
                                continue
                            end

                            seed = [ (px + rand() - 1) / scale + 1, (py + rand() - 1) / scale + 1 ]

                            # compute pixels in the path around the seed point

                            pixels = Deque{Tuple{Int64, Int64}}()
                            push!(pixels, (px, py))

                            # Compute initial eigenvectors for seeding

                            tRoot = interpolate(tf, seed)
                            evecRoot = vector(tRoot,asymmetric)

                            # forward pass
                            num_forward = 0
                            xf = seed

                            vf = evecRoot
                            for step in 1:num_steps
                                vf = rk4_vector(tf, xf, δ, vf, asymmetric, evecScale)
                                xf = xf + vf
                                if !inBounds(tf, xf)
                                    break
                                end
                                push!(pixels, to_pixel(xf,scale))
                                num_forward += 1
                                # if pixel_near(xf, wedge_pixels)
                                #     break
                                # end
                            end

                            # backward pass
                            num_backward = 0
                            xb = seed
                            vb = -evecRoot
                            for step in 1:num_steps
                                vb = rk4_vector(tf, xb, δ, vb, asymmetric, evecScale)
                                xb = xb + vb
                                if !inBounds(tf, xb)
                                    break
                                end
                                pushfirst!(pixels, to_pixel(xb,scale))
                                # if pixel_near(first(pixels), wedge_pixels)
                                #     break
                                # end
                                num_backward += 1
                            end
                            
                            intensity = 0

                            for pixel in pixels
                                intensity += noise[pixel[1],pixel[2]]
                            end

                            finalImage[ px, py ] += intensity / length(pixels)
                            numHits[ px, py ] += 1
                        
                            # propagate intensities forward

                            pixels_f = collect(pixels)
                            head_f = num_forward # the number of pixels in front of the one that we are currently working with.
                            tail_f = num_backward # the number of pixels behind the one that we are currently working with.
                            tail_f_position = 1
                            intensity_f = intensity
                            steps = 0 # kill counter to avoid getting stuck in a loop

                            if num_forward < num_steps
                                at_edge = true
                            else
                                at_edge = false
                            end

                            index = 0
                            for pixel in pixels_f
                                index += 1
                                if index <= num_backward
                                    continue
                                end

                                steps += 1
                                if steps >= max_path_tracing
                                    break
                                end

                                if !at_edge
                                    vf = rk4_vector(tf, xf, δ, vf, asymmetric, evecScale)
                                    xf = xf + vf
                                    if !inBounds(tf, xf)
                                        at_edge = true
                                    end
                                end

                                if tail_f < num_steps
                                    tail_f += 1
                                else
                                    furthestTail = pixels_f[tail_f_position]
                                    intensity_f -= noise[ furthestTail[1], furthestTail[2] ]
                                    tail_f_position += 1
                                end

                                if at_edge
                                    head_f -= 1
                                else
                                    nextPixel = to_pixel(xf,scale)
                                    push!(pixels_f, nextPixel)
                                    intensity_f += noise[ nextPixel[1], nextPixel[2] ]
                                    # if pixel_near(nextPixel, wedge_pixels)
                                    #     at_edge = true
                                    # end
                                end

                                finalImage[ pixel[1], pixel[2] ] += intensity_f / ( head_f + tail_f + 1 )
                                numHits[ pixel[1], pixel[2] ] += 1
                            end

                            # propagate intensities backward

                            pixels_b = reverse(collect(pixels))
                            head_b = num_backward # the number of pixels in front of the one that we are currently working with.
                            tail_b = num_forward # the number of pixels behind the one that we are currently working with.
                            tail_b_position = 1
                            intensity_b = intensity

                            if num_backward < num_steps
                                at_edge = true
                            else
                                at_edge = false
                            end

                            index = 0
                            for pixel in pixels_b
                                index += 1
                                if index <= num_forward
                                    continue
                                end

                                steps += 1
                                if steps >= max_path_tracing
                                    break
                                end

                                if !at_edge
                                    vb = rk4_vector(tf, xb, δ, vb, asymmetric, evecScale)
                                    xb = xb + vb
                                    if !inBounds(tf, xb)
                                        at_edge = true
                                    end
                                end

                                if tail_b < num_steps
                                    tail_b += 1
                                else
                                    furthestTail = pixels_b[tail_b_position]
                                    intensity_b -= noise[ furthestTail[1], furthestTail[2] ]
                                    tail_b_position += 1
                                end

                                if at_edge
                                    head_b -= 1
                                else
                                    nextPixel = to_pixel(xb,scale)
                                    push!(pixels_b, nextPixel)
                                    intensity_b += noise[ nextPixel[1], nextPixel[2] ]
                                    # if pixel_near(nextPixel, wedge_pixels)
                                    #     at_edge = true
                                    # end
                                end

                                finalImage[ pixel[1], pixel[2] ] += intensity_b / ( head_b + tail_b + 1 )
                                numHits[ pixel[1], pixel[2] ] += 1

                            end

                        end # end if px and py are valid coordinates

                    end # end for bx in 1:blocks_dims[1]
                end # end for by in 1:blocks_dims[2]
            end # end for x in 1:block_size
        end # end for y in 1:block_size

        for j in 1:imageSize[2]
            for i in 1:imageSize[1]
                finalImage[i,j] /= numHits[i,j]
            end
        end
    end

    t2 = time()
    print(t2-t1)
    return

    cmap_list = [(0.0,"#4155c8"), (1.0,"#a1c0ff")]
    cmap = colors.LinearSegmentedColormap.from_list("custom_blue", cmap_list)

    if save_lic != ""
        outf = open(save_lic, "w")
        write(outf, finalImage)
        close(outf)
    end

    # visualize
    finalImage = finalImage .^ power
    plt.imshow(transpose(finalImage), cmap=cmap)

    for cp in trisector_points
        plt.scatter( (cp[1]-1)*scale-0.5, (cp[2]-1)*scale-0.5, color="#eaeaea", s=500 )
        plt.scatter( (cp[1]-1)*scale-0.5, (cp[2]-1)*scale-0.5, color="#000000", s=300 )
    end

    for cp in wedge_points
        plt.scatter( (cp[1]-1)*scale-0.5, (cp[2]-1)*scale-0.5, color="#000000", s=450 )
        plt.scatter( (cp[1]-1)*scale-0.5, (cp[2]-1)*scale-0.5, color="#eaeaea", s=300 )
    end

    for cp in trisector_points_alt
        # plt.scatter( (cp[1]-1)*scale-0.5, (cp[2]-1)*scale-0.5, color="#000000", s=500, alpha=0.6 )
        plt.scatter( (cp[1]-1)*scale-0.5, (cp[2]-1)*scale-0.5, color="#c9a8f5", s=500, alpha=0.7 )
    end

    for cp in wedge_points_alt
        # plt.scatter( (cp[1]-1)*scale-0.5, (cp[2]-1)*scale-0.5, color="#000000", s=450, alpha=0.6 )
        plt.scatter( (cp[1]-1)*scale-0.5, (cp[2]-1)*scale-0.5, color="#e08e45", s=500, alpha=0.7 )
    end

    # uncomment to show cell boundaries (not recommended lol)
    # for j in 1:size[2]
    #     for i in 1:size[1]-1
    #         plt.plot( [ (i-1)*scale-0.5, i*scale-0.5 ], [(j-1)*scale-0.5,(j-1)*scale-0.5], color="black" )
    #     end
    # end

    # for j in 1:size[2]-1
    #     for i in 1:size[1]
    #         plt.plot( [(i-1)*scale-0.5, (i-1)*scale-0.5], [(j-1)*scale-0.5, j*scale-0.5], color="black" )
    #     end
    # end

    # for j in 1:size[2]-1
    #     for i in 1:size[1]-1
    #         plt.plot( [(i-1)*scale-0.5, i*scale-0.5], [j*scale-0.5, (j-1)*scale-0.5], color="black" )
    #     end
    # end

    # plt.show()
end

# main()