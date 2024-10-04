using PyCall
using LinearAlgebra

plt = pyimport("matplotlib.pyplot")
lic = pyimport("lic")

folder = "../data/fakeSym/wind1"
size = (200,100)
scale = 8
length=30

a = time()
a_file = open("$folder/row_1_col_1.dat", "r")
b_file = open("$folder/row_1_col_2.dat", "r")
c_file = open("$folder/row_2_col_1.dat", "r")
d_file = open("$folder/row_2_col_2.dat", "r")

a_array = reshape( reinterpret( Float64, read(a_file) ), size )
b_array = reshape( reinterpret( Float64, read(b_file) ), size )
c_array = reshape( reinterpret( Float64, read(c_file) ), size )
d_array = reshape( reinterpret( Float64, read(d_file) ), size )

x = zeros( ((size[1]-1)*scale+1, (size[2]-1)*scale+1) )
y = zeros( ((size[1]-1)*scale+1, (size[2]-1)*scale+1) )

# barycentric coordinates coefficients
bcc = zeros( Float64, (scale, scale, 4) )
for j in 1:scale
    for i in 1:scale
        if i + j <= scale + 1
            bcc[i,j,1] = (j-1) / (scale)
            bcc[i,j,2] = 0
            bcc[i,j,3] = (i-1) / (scale)
            bcc[i,j,4] = ( scale - i - j + 2 ) / (scale)
        else
            bcc[i,j,1] = (scale - i + 1) / (scale)
            bcc[i,j,2] = (i + j - 2 - scale) / (scale)
            bcc[i,j,3] = (scale - j + 1) / (scale)
            bcc[i,j,4] = 0
        end
    end
end

a_large = zeros(Float64, ( (size[1]-1)*scale+1, (size[2]-1)*scale+1 ))
b_large = zeros(Float64, ( (size[1]-1)*scale+1, (size[2]-1)*scale+1 ))
d_large = zeros(Float64, ( (size[1]-1)*scale+1, (size[2]-1)*scale+1 ))

# interpolate
for j in 1:(size[2]-1)*scale + 1
    for i in 1:(size[1]-1)*scale + 1

        if i != (size[1]-1)*scale + 1
            x1, x2, x3, x4 = (i-1) ÷ scale + 1, (i-1) ÷ scale + 2, (i-1) ÷ scale + 2, (i-1) ÷ scale + 1
        else
            x1, x2, x3, x4 = (i-1) ÷ scale + 1, (i-1) ÷ scale + 1, (i-1) ÷ scale + 1, (i-1) ÷ scale + 1
        end

        if j != (size[2]-1)*scale + 1
            y1, y2, y3, y4 = (j-1) ÷ scale + 2, (j-1) ÷ scale + 2, (j-1) ÷ scale + 1, (j-1) ÷ scale + 1
        else
            y1, y2, y3, y4 = (j-1) ÷ scale + 1, (j-1) ÷ scale + 1, (j-1) ÷ scale + 1, (j-1) ÷ scale + 1
        end

        c1, c2, c3, c4 = bcc[ (i-1)%scale + 1, (j-1)%scale + 1, : ]

        interp = [
            c1 * a_array[x1,y1] + c2 * a_array[x2,y2] + c3 * a_array[x3,y3] + c4 * a_array[x4,y4] [
            c1 * b_array[x1,y1] + c2 * b_array[x2,y2] + c3 * b_array[x3,y3] + c4 * b_array[x4,y4] ] ;
            c1 * c_array[x1,y1] + c2 * c_array[x2,y2] + c3 * c_array[x3,y3] + c4 * c_array[x4,y4] [
            c1 * d_array[x1,y1] + c2 * d_array[x2,y2] + c3 * d_array[x3,y3] + c4 * d_array[x4,y4] ]
        ]

        a_large[i,j] = interp[1,1]
        b_large[i,j] = interp[1,2]
        d_large[i,j] = interp[2,2]

        e = eigvecs(interp)[:,2]
        if e[1] < 0
            e *= -1
        end

        x[i,j] = e[1]
        y[i,j] = e[2]

    end
end

trisector_x = []
trisector_y = []
wedge_x = []
wedge_y = []

for j in 1:(size[2]-1)*scale
    for i in 1:(size[1]-1)*scale
        for k in 0:1

            top = Bool(k)

            if top
                x1, x2, x3 = i+1, i, i+1
                y1, y2, y3 = j+1, j+1, j
            else
                x1, x2, x3 = i, i+1, i
                y1, y2, y3 = j, j, j+1
            end

            s12 = b_large[x2,y2]*(d_large[x1,y1] - a_large[x1,y1]) - b_large[x1,y1]*(d_large[x2,y2] - a_large[x2,y2])
            s23 = b_large[x3,y3]*(d_large[x2,y2] - a_large[x2,y2]) - b_large[x2,y2]*(d_large[x3,y3] - a_large[x3,y3])
            s31 = b_large[x1,y1]*(d_large[x3,y3] - a_large[x3,y3]) - b_large[x3,y3]*(d_large[x1,y1] - a_large[x1,y1])

            if s12 > 0 && s23 > 0 && s31 > 0
                
                if top
                    push!(wedge_x, i + 2/3)
                    push!(wedge_y, j + 2/3)
                else
                    push!(wedge_x, i + 1/2)
                    push!(wedge_y, j + 1/3)
                end

            elseif s12 < 0 && s23 < 0 && s31 < 0

                if top
                    push!(trisector_x, i + 2/3)
                    push!(trisector_y, j + 2/3)
                else
                    push!(trisector_x, i + 1/2)
                    push!(trisector_y, j + 1/3)
                end

            end


        end
    end
end

b = time()

licplot = lic.lic(transpose(x), transpose(y), length=length)

plt.imshow(licplot, cmap="gray")

plt.scatter(trisector_x,trisector_y,color="black", s=1000)
plt.scatter(wedge_x, wedge_y, color="white", s=1000)

plt.scatter(trisector_x,trisector_y,color="white", s=800)
plt.scatter(wedge_x, wedge_y, color="black", s=800)

plt.show()