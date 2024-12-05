module tensorField

using LinearAlgebra
using StaticArrays
using ..utils

export TensorField2d

export loadTensorField2dFromFolder
export getTensor
export setTensor
export getTensorsAtCell
export getCircularPointType
export deviator
export decomposeTensor
export recomposeTensor
export decomposeTensorSymmetric
export edgesMatch
export recomposeTensorSymmetric
export classifyTensorEigenvector
export classifyTensorEigenvalue
export classifyEdge
export zeroTensorField

export saveTensorField32
export saveTensorField64
export saveSymmetricTensorField32
export saveSymmetricTensorField64

struct TensorField2d
    A::Array{Float64}
    B::Array{Float64}
    C::Array{Float64}
    D::Array{Float64}
    dims::Tuple{Int64, Int64, Int64}
end

function zeroTensorField(dims)
    return TensorField2d64(zeros(Float64,dims),zeros(Float64,dims),zeros(Float64,dims),zeros(Float64,dims), dims)
end

function loadTensorField2dFromFolder(folder::String, dims::Tuple{Int64, Int64, Int64})

    num_entries = dims[1]*dims[2]*dims[3]
    num_bytes = filesize("$folder/row_1_col_1.dat")/num_entries

    if num_bytes == 8
        A_byte_file = open("$folder/row_1_col_1.dat", "r")
        A = reshape( reinterpret( Float64, read(A_byte_file) ), dims )
        close(A_byte_file)
    
        B_byte_file = open("$folder/row_1_col_2.dat", "r")
        B = reshape( reinterpret( Float64, read(B_byte_file) ), dims )
        close(B_byte_file)
    
        C_byte_file = open("$folder/row_2_col_1.dat", "r")
        C = reshape( reinterpret( Float64, read(C_byte_file) ), dims )
        close(C_byte_file)
    
        D_byte_file = open("$folder/row_2_col_2.dat", "r")
        D = reshape( reinterpret( Float64, read(D_byte_file) ), dims )
        close(D_byte_file)
    elseif num_bytes == 4
        println("loading from 32 bits to 64")
        A_byte_file = open("$folder/row_1_col_1.dat", "r")
        A = Array{Float64}(reshape( reinterpret( Float32, read(A_byte_file) ), dims ))
        close(A_byte_file)
    
        B_byte_file = open("$folder/row_1_col_2.dat", "r")
        B = Array{Float64}(reshape( reinterpret( Float32, read(B_byte_file) ), dims ))
        close(B_byte_file)
    
        C_byte_file = open("$folder/row_2_col_1.dat", "r")
        C = Array{Float64}(reshape( reinterpret( Float32, read(C_byte_file) ), dims ))
        close(C_byte_file)
    
        D_byte_file = open("$folder/row_2_col_2.dat", "r")
        D = Array{Float64}(reshape( reinterpret( Float32, read(D_byte_file) ), dims ))
        close(D_byte_file)
    else
        println("The file that you specified is $num_bytes bytes per point")
        println("only 32 and 64 bits are currently supported")        
        exit(1)
    end

    tf = TensorField2d(A,B,C,D,dims)

    return tf

end

function saveTensorField64(folder::String, tf::TensorField2d, suffix::String="")
    saveArray64("$folder/row_1_col_1$suffix.dat", tf.A)
    saveArray64("$folder/row_1_col_2$suffix.dat", tf.B)
    saveArray64("$folder/row_2_col_1$suffix.dat", tf.C)
    saveArray64("$folder/row_2_col_2$suffix.dat", tf.D)
end

function saveSymmetricTensorField64(folder::String, tf::TensorField2d, suffix::String="")
    saveArray64("$folder/row_1_col_1$suffix.dat", tf.A)
    saveArray64("$folder/row_1_col_2$suffix.dat", tf.B)
    saveArray64("$folder/row_2_col_2$suffix.dat", tf.D)
end

function saveTensorField32(folder::String, tf::TensorField2d, suffix::String="")
    saveArray32("$folder/row_1_col_1$suffix.dat", tf.A)
    saveArray32("$folder/row_1_col_2$suffix.dat", tf.B)
    saveArray32("$folder/row_2_col_1$suffix.dat", tf.C)
    saveArray32("$folder/row_2_col_2$suffix.dat", tf.D)
end

function saveSymmetricTensorField32(folder::String, tf::TensorField2d, suffix::String="")
    saveArray32("$folder/row_1_col_1$suffix.dat", tf.A)
    saveArray32("$folder/row_1_col_2$suffix.dat", tf.B)
    saveArray32("$folder/row_2_col_2$suffix.dat", tf.D)
end

function getTensor(tf::TensorField2d, x::Int64, y::Int64, t::Int64)
    return SMatrix{2,2,Float64}( tf.A[x,y,t], tf.C[x,y,t], tf.B[x,y,t], tf.D[x,y,t] )
end

function setTensor(tf::TensorField2d, x, y, t, tensor::FloatMatrix)
    tf.A[x,y,t] = tensor[1,1]
    tf.B[x,y,t] = tensor[1,2]
    tf.C[x,y,t] = tensor[2,1]
    tf.D[x,y,t] = tensor[2,2]
end

# returns in counterclockwise orientation, consistent with getCellVertexCoords
function getTensorsAtCell(tf::TensorField2d, x::Int64, y::Int64, t::Int64, top::Bool)
    points = getCellVertexCoords(x,y,t,top)
    return ( getTensor(tf, points[1]...), getTensor(tf, points[2]...), getTensor(tf, points[3]...) )
end

function getCircularPointType( tf::TensorField2d, x::Int64, y::Int64, t::Int64, top::Bool )
    return getCircularPointType( getTensorsAtCell( tf, x, y, t, top )... )
end

function deviator(tensor::FloatMatrix)
    return tensor - 0.5*tr(tensor)*I
end

function symmetricDeviator(tensor::FloatMatrix)
    diagonal = 0.5*tr(tensor)
    off_diagonal = 0.5*(tensor[2,1]-tensor[1,2])
    return SMatrix{2,2,Float64}( tensor[1,1] - diagonal, tensor[2,1] - off_diagonal, tensor[1,2] + off_diagonal, tensor[2,2] - diagonal )
    # return tensor - 0.5*tr(tensor)*I + 0.5*(tensor[2,1]-tensor[1,2])*[ 0 1 ; -1 0 ]
end

# we assume that the tensors are in counterclockwise direction
function getCircularPointType(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix, verbose=false)
    # rather than explicitly computing the deviator it is faster to do it this way.
    # yes I know the readability kind of sucks but it makes kind of a huge difference.
    D1_11 = tensor1[1,1] - tensor1[2,2]
    D1_21 = tensor1[2,1] + tensor1[1,2]

    D2_11 = tensor2[1,1] - tensor2[2,2]
    D2_21 = tensor2[2,1] + tensor2[1,2]

    D3_11 = tensor3[1,1] - tensor3[2,2]
    D3_21 = tensor3[2,1] + tensor3[1,2]

    sign1 = sign(D1_11*D2_21 - D2_11*D1_21)
    sign2 = sign(D2_11*D3_21 - D3_11*D2_21)
    sign3 = sign(D3_11*D1_21 - D1_11*D3_21)

    if sign1 == 0 || sign2 == 0 || sign3 == 0
        return CP_ERROR
    end

    if sign1 == sign2
        if sign3 == sign1
            if sign3 == 1
                return CP_WEDGE
            else
                return CP_TRISECTOR
            end
        else
            return CP_NORMAL
        end
    else
        return CP_NORMAL
    end
end

# we assume that the tensors are in counterclockwise direction
function getCircularPointType(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix, verbose=false)
    D1 = symmetricDeviator(tensor1)
    D2 = symmetricDeviator(tensor2)
    D3 = symmetricDeviator(tensor3)

    sign1 = sign(D1[1,1]*D2[2,1] - D2[1,1]*D1[2,1])
    sign2 = sign(D2[1,1]*D3[2,1] - D3[1,1]*D2[2,1])
    sign3 = sign(D3[1,1]*D1[2,1] - D1[1,1]*D3[2,1])

    if sign1 == 0 || sign2 == 0 || sign3 == 0
        return CP_ERROR
    end

    if sign1 == sign2
        if sign3 == sign1
            if sign3 == 1
                return CP_WEDGE
            else
                return CP_TRISECTOR
            end
        else
            return CP_NORMAL
        end
    else
        return CP_NORMAL
    end
end

function decomposeTensor(tensor::FloatMatrix)

    y_d::Float64 = (tensor[1,1] + tensor[2,2])/2
    y_r::Float64 = (tensor[2,1] - tensor[1,2])/2
    # tensor -= [ y_d -y_r ; y_r y_d ]

    # cplx = tensor[1,1] + tensor[1,2]*im
    cplx = (tensor[1,1] - y_d) + (tensor[1,2]+y_r)*im
    y_s::Float64 = abs(cplx)
    θ::Float64 = angle(cplx)

    return (y_d, y_r, y_s, θ)
end

function recomposeTensor(y_d::AbstractFloat, y_r::AbstractFloat, y_s::AbstractFloat, θ::AbstractFloat)

    cos_ = cos(θ)
    sin_ = sin(θ)

    return SMatrix{2,2,Float64}( y_d+y_s*cos_,  y_r+y_s*sin_,  -y_r+y_s*sin_,  y_d-y_s*cos_ )

end

function decomposeTensorSymmetric(T::SMatrix{2,2,Float64,4})
    trace = (T[1,1] + T[2,2]) / 2
    cplx = (T[1,1] - T[2,2])/2 + T[1,2]*im
    r = abs(cplx)
    θ = angle(cplx)
    return (trace,r,θ)
end

function recomposeTensorSymmetric(trace,r,θ)
    return SMatrix{2,2,Float64}( trace + r*cos(θ), r*sin(θ), r*sin(θ), trace - r*cos(θ) )
end

function classifyTensorEigenvector(yr::AbstractFloat, ys::AbstractFloat)
    if abs(yr) < ϵ
        return SYMMETRIC
    elseif yr > 0
        if abs(yr-ys) < ϵ
            return PI_BY_4
        elseif ys > yr
            return W_RN
        else
            return W_CN
        end
    else
        if abs(yr+ys) < ϵ
            return MINUS_PI_BY_4
        elseif ys > -yr
            return W_RS
        else
            return W_CS
        end
    end
end

function classifyTensorEigenvector(tensor::FloatMatrix)
    _, yr, ys, _ = decomposeTensor(tensor)

    return classifyTensorEigenvector(yr, ys)
end

function classifyTensorEigenvalue(tensor::FloatMatrix)
    yd, yr, ys, _ = decomposeTensor(tensor)
    
    return classifyTensorEigenvalue(yd,yr,ys)
end

function classifyTensorEigenvalue(yd::AbstractFloat, yr::AbstractFloat, ys::AbstractFloat)

    if ys > abs(yd) && ys > abs(yr)
        return ANISOTROPIC_STRETCHING
    elseif abs(yr) > abs(yd)
        if yr > 0
            return COUNTERCLOCKWISE_ROTATION
        else
            return CLOCKWISE_ROTATION
        end
    else
        if yd > 0
            return POSITIVE_SCALING
        else
            return NEGATIVE_SCALING
        end
    end

end

function edgesMatch( t11::FloatMatrix, t21::FloatMatrix, t12::FloatMatrix, t22::FloatMatrix, eb::Float64 )
    if t11 == t12 && t21 == t22
        return true
    end

    evalClass1, evalLoc1, evecClass1, evecLoc1 = classifyEdgeOuter( t11, t21 )
    evalClass2, evalLoc2, evecClass2, evecLoc2 = classifyEdgeOuter( t12, t22 )

    if length(evalClass1) > 0 && evalClass1[1] == 99 || length(evalClass2) > 0 && evalClass2[1] == 99
        return false
    end

    return evalClass1 == evalClass2 && evecClass1 == evecClass2 && (length(evalClass1) == 0 || maximum( abs.(evalLoc1-evalLoc2) ) <= eb) && (evecClass1 == 0 || maximum( abs.(evecLoc1-evecLoc2) ) <= eb )
end

function classifyEdgeOuter( t1::FloatMatrix, t2::FloatMatrix, p=false )
    decomp1 = decomposeTensor(t1)
    decomp2 = decomposeTensor(t2)
    return classifyEdge(decomp1..., decomp2...,p)
end

# returns a list of numbers with the following entries (ordered)
# as one moves from t2 -> t1 (small to large interp values)
# values are -2 signifying r=-d,
#            -1 signifying r=d
#             0 signifying anisotropic stretching starts or ends being dominant
# these correspond to vertex patterns from the visualization.
# The fourth number corresponds to the bin number when using the bin method.
function classifyEdge( d1::AbstractFloat, r1::AbstractFloat, s1::AbstractFloat, θ1::AbstractFloat, d2::AbstractFloat, r2::AbstractFloat, s2::AbstractFloat, θ2::AbstractFloat,p=false )

    y1 = (d2-r2) / ( (d2-r2) - (d1 - r1) )
    y2 = (d2+r2) / ( (d2+r2) - (d1 + r1) )

    if 0 <= y1 <= 1
        if 0 <= y2 <= 1
            if y1 < y2
                cross_values = [(-1, y1, 0), (-2,y2,0)]
            else
                cross_values = [(-2,y2, 0), (-1,y1,0)]
            end
        else
            cross_values = [(-1,y1,0)]
        end
    else
        if 0 <= y2 <= 1
            cross_values = [(-2,y2,0)]
        else
            cross_values::Vector{Tuple{Int64,Float64,Int64}} = Vector{Tuple{Int64, Float64, Int64}}(undef,0)
        end
    end

    u_base = s1^2
    v_base = 2*s1*s2*cos(θ1 - θ2)
    w_base = s2^2

    # we will now classify two parabolas whose values correspond to s^2-r^2 and s^2-d^2, respectively,
    # in the interval [0,1]. The behavior of their sign changes will determine if/where s dominates the other
    # two coefficients.

    # 0 - always positive
    # 1 - always negative
    # 2 - positive -> negative
    # 3 - negative -> positive
    # 4 - positive -> negative -> positive
    # 5 - negative -> positive -> negative
    intercept_categories = [-1, -1]
    c1_list = [d1, r1]
    c2_list = [d2, r2]
    
    for i in 1:2
        c1 = c1_list[i]
        c2 = c2_list[i]

        # coefficients for ut^2 + vt(1-t) + w(1-t)^2. As t->[0,1] this goes from w to u
        u = u_base - c1^2
        v = v_base - 2*c1*c2
        w = w_base - c2^2

        su = sign(u)
        sw = sign(w)

        # if su < 0.0001 * ϵ
        #     su = 0
        # end

        # if sw < 0.0001 * ϵ
        #     sw = 0
        # end

        if abs(u) < 0.00000001*ϵ
            su = 0.0
            u = 0.0
        end

        if abs(w) < 0.00000001*ϵ
            sw = 0.0
            w = 0.0
        end

        if p
            println(("vals",su,sw,u,w))
        end

        if su == 0 || sw == 0
            # Degenerate case
            if su == sw
                if v > 0
                    category = 0
                else
                    # Always 0 is the same as always negative due to how ties are broken.
                    category = 1
                end
            elseif su == 0
                if v < -0.0001*ϵ && w > 0
                    category = 2
                elseif v > 0.0001*ϵ && w < 0
                    category = 3
                elseif w > 0
                    category = 0
                else
                    category = 1
                end
            else
                if u < 0 && v > 0.0001*ϵ
                    category = 2
                elseif u > 0 && v < -0.0001*ϵ
                    category = 3
                elseif u > 0
                    category = 0
                else
                    category = 1
                end
            end
        else
            # Nondegenerate case
            if su != sw
                if sw > 0
                    category = 2
                else
                    category = 3
                end
            else
                sv = sign(v)
                if sv == -su && (v^2 - 4*u*w) > 0
                    if su > 0
                        category = 4
                    else
                        category = 5
                    end
                else
                    if su > 0
                        category = 0
                    else
                        category = 1
                    end
                end
            end
        end

        # Now store important markers in cross_values as a tuple
            # 1st value: 0 (to distinguish from the other crossing types)
            # 2nd value: interp value in [0,1]
            # 3rd value: 1 for d, 2 for r

        if category != 0 && category != 1
            # # scale up the values to remove numerical instability:
            # if u < 10e-4 && v < 10e-4 && w < 10e-4
            #     u *= 10e5
            #     v *= 10e5
            #     w *= 10e5
            # end

            square_root = sqrt(v^2-4*u*w)
            t1 = (2*w-v + square_root) / (2*u - 2*v + 2*w)
            t2 = (2*w-v - square_root) / (2*u - 2*v + 2*w)

            # We run into numerical precision issues during the computation so raise the precision...
            if abs(t1) > 10e12 || abs(t2) > 10e12
                return ([99], [0.0], 0, [0.0])
            end

            if p
                println("before interp: $((u,v,w))")
            end

            small_t = round(min(t1, t2), digits=14)
            large_t = round(max(t1, t2), digits=14)

            if p
                println("t values",(small_t,large_t))
            end

            if -0.0001*ϵ <= small_t < 0.0
                small_t = 0.0
            end

            if -0.0001*ϵ <= large_t < 0.0
                large_t = 0.0
            end

            if 1.0 < small_t < 1.0 + 0.0001*ϵ
                small_t = 1.0
            end

            if 1.0 < large_t < 1.0 + 0.0001*ϵ
                large_t = 1.0
            end

            if category == 2 || category == 3

                if su == 0
                    push!(cross_values, (0, small_t, i))
                elseif sw == 0
                    push!(cross_values, (0, large_t, i))
                elseif 0 <= small_t <= 1
                    push!(cross_values, (0, small_t, i))
                else
                    push!(cross_values, (0, large_t, i))
                end
            elseif category == 4 || category == 5
                push!(cross_values, (0, small_t, i))
                push!(cross_values, (0, large_t, i))
            end
        end

        intercept_categories[i] = category
    end


    # from the array of cross values, generate the edge class
    sort!(cross_values, by=f(x)=(x[2],x[1],x[3]))
    eigenvalueEdgeClass::Vector{Int64} = Vector{Float64}(undef, 0)
    eigenvalueEdgeLocations::Vector{Float64} = Vector{Float64}(undef, 0)
    eigenvectorEdgeClass = 0
    eigenvectorEdgeLocations::Vector{Float64} = Vector{Float64}(undef, 0)
    
    # 1st entry - s is larger than d
    # 2nd entry - s is larger than r
    s_is_larger = [ intercept_categories[1] % 2 == 0, intercept_categories[2] % 2 == 0 ]

    if p
        println(s_is_larger)
        println(cross_values)
        println(intercept_categories)
    end

    for i in eachindex(cross_values)

        c = cross_values[i]

        if c[1] == 0.0

            if c[2] > 0.0001*ϵ && c[2] < 1-0.0001*ϵ
                if (c[3] == 1 || s_is_larger[1]) && (c[3] == 2 || s_is_larger[2])
                    push!(eigenvalueEdgeClass, 0)
                    push!(eigenvalueEdgeLocations, c[2])
                end
                if c[3] == 2
                    eigenvectorEdgeClass += 1
                    push!(eigenvectorEdgeLocations, c[2])
                end
                s_is_larger[c[3]] ⊻= true
            end   

        elseif !( s_is_larger[1] && s_is_larger[2] ) && c[2] > 0.0001*ϵ && c[2] < 1.0-0.0001*ϵ && ( i == length(cross_values) || cross_values[i+1][2] != c[2] )
            # The last conditional is to handle the degenerate case where |r|=|d| crosses at the same time as |r|=s or |d|=s.
            # In that case, we ignore the swap here as it is a switch between two colors where one will be hidden anyway.
            push!(eigenvalueEdgeClass, c[1])
            push!(eigenvalueEdgeLocations, c[2])
        end

    end

    return eigenvalueEdgeClass, eigenvalueEdgeLocations, eigenvectorEdgeClass, eigenvectorEdgeLocations

end

# returns a list of numbers with the following entries (ordered)
# as one moves from t2 -> t1 (small to large interp values)
# values are -1 signifying r=-d,
#             1 signifying r=d
#             0 signifying anisotropic stretching starts or ends being dominant
# these correspond to vertex patterns from the visualization.
function classifyEdgeEigenvalueOld( t1::FloatMatrix, t2::FloatMatrix )

    d1, r1, s1, θ1 = decomposeTensor(t1)
    d2, r2, s2, θ2 = decomposeTensor(t2)

    # point where r=d
    y1 = (d2-r2) / ( (d2-r2) - (d1 - r1) )

    # point where r=-d
    y2 = (d2+r2) / ( (d2+r2) - (d1 + r1) )

    # points where r=|s| or r=|d|

    # as = anisotropic stretching
    as_values = []

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
                        push!(as_values, c)
                    end
                end
            end
        end
    end

    if length(as_values) >= 2
        value_candidates = [(y1, 1), (y2, -1), (minimum(as_values), 0), (maximum(as_values), 0)]
    elseif length(as_values) == 1
        value_candidates = [(y1, 1), (y2, -1), (as_values[1], 0)]
    else
        value_candidates = [(y1, 1), (y2, -1)]
    end

    values = []

    # portion of list corresponding to anisotropic stretching
    if s2 >= d2 && s2 >= r2
        in_as = true
    else
        in_as = false
    end
    
    for v in value_candidates
        if (v[2] == 1 || v[2] == -1) && (0 <= v[1] <= 1) && !in_as
            push!(values, v[2])
        elseif v[2] == 0
            push!(values, v[2])
            in_as ⊻= true
        end
    end

    return values

end

end