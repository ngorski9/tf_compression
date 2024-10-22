module tensorField

using LinearAlgebra
using ..utils

export TensorField2d
export SymmetricTensorField2d
export SymmetricTensorField2d64

export loadTensorField2dFromFolder
export getTensor
export setTensor
export getTensorsAtCell
export getCircularPointType
export deviator
export decomposeTensor
export recomposeTensor
export decomposeTensorSymmetric
export recomposeTensorSymmetric
export classifyTensorEigenvector
export classifyTensorEigenvalue
export classifyEdgeEigenvalue
export classifyEdgeEigenvalueOld
export zeroTensorField

abstract type TensorField2d end

abstract type SymmetricTensorField2d <: TensorField2d end
abstract type AsymmetricTensorField2d <: TensorField2d end

# First axis is the time axis
mutable struct SymmetricTensorField2d64 <: SymmetricTensorField2d
    entries::Array{Array{Float64}}
    dims::Tuple{Int64, Int64, Int64}
end

mutable struct TensorField2d64 <: AsymmetricTensorField2d 
    entries::Array{Array{Float64}}
    dims::Tuple{Int64, Int64, Int64}
end

function zeroTensorField(dtype, dims)
    entries = Matrix{Array{dtype}}(undef, (2,2))
    for row in 1:2
        for col in 1:2
            entries[row,col] = zeros(dtype, dims)
        end
    end

    if dtype == Float64
        return TensorField2d64(entries, dims)
    end
end

function loadTensorField2dFromFolder(folder::String, dims::Tuple{Int64, Int64, Int64})

    num_entries = dims[1]*dims[2]*dims[3]
    num_bytes = filesize("$folder/row_1_col_1.dat")/num_entries

    if num_bytes == 8
        entries = Array{Array{Float64}}(undef, (2,2))
        dtype = Float64
    elseif num_bytes == 4
        entries = Array{Array{Float64}}(undef, (2,2))
        println("loading from 32 bits to 64")
        dtype = Float32
        num_bytes = 8
    else
        println("The file that you specified is $num_bytes bytes per point")
        println("only 32 and 64 bits are currently supported")
        exit(1)
    end

    for row in 1:2
        for col in 1:2
            byte_file = open("$folder/row_$(row)_col_$(col).dat", "r")
            arr = reshape( reinterpret( dtype, read(byte_file)), dims ) 
            entries[row,col] = arr
        end
    end

    if dtype == Float32
        dtype = Float64
    end

    sym = true
    for i in 1:num_entries
        if abs(entries[1,2][i] - entries[2,1][i]) > ϵ
            sym = false
            break
        end
    end

    if sym
        if num_bytes == 8
            tf = SymmetricTensorField2d64(entries, dims)
        end
    else
        if num_bytes == 8
            tf = TensorField2d64(entries, dims)
        end
    end

    return tf, dtype

end

function getTensor(tf::TensorField2d, t::Int64, x::Int64, y::Int64)
    return [ tf.entries[1,1][t,x,y] tf.entries[1,2][t,x,y] ; tf.entries[2,1][t,x,y] tf.entries[2,2][t,x,y] ]
end

function setTensor(tf::TensorField2d, t, x, y, tensor::FloatMatrix)
    for row in 1:2
        for col in 1:2
            tf.entries[row, col][t,x,y] = tensor[row,col]
        end
    end
end

# returns in counterclockwise orientation, consistent with getCellVertexCoords
function getTensorsAtCell(tf::TensorField2d, t::Int64, x::Int64, y::Int64, top::Bool)
    points = getCellVertexCoords(t,x,y,top)
    return [ getTensor(tf, points[1]...), getTensor(tf, points[2]...), getTensor(tf, points[3]...) ]
end

function getCircularPointType( tf::TensorField2d, t::Int64, x::Int64, y::Int64, top::Bool )
    return getCircularPointType( getTensorsAtCell( tf, t, x, y, top )... )
end

function deviator(tensor::FloatMatrix)
    return tensor - 0.5*tr(tensor)*I
end

function symmetricDeviator(tensor::FloatMatrix)
    return tensor - 0.5*tr(tensor)*I + 0.5*(tensor[2,1]-tensor[1,2])*[ 0 1 ; -1 0 ]
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
    tensor -= [ y_d -y_r ; y_r y_d ]

    cplx = tensor[1,1] + tensor[1,2]*im
    y_s::Float64 = abs(cplx)
    θ::Float64 = angle(cplx)

    return (y_d, y_r, y_s, θ)
end

function recomposeTensor(y_d::AbstractFloat, y_r::AbstractFloat, y_s::AbstractFloat, θ::AbstractFloat)

    cos_ = cos(θ)
    sin_ = sin(θ)

    return[ y_d+y_s*cos_ -y_r+y_s*sin_ ; y_r+y_s*sin_ y_d-y_s*cos_ ]

end

function decomposeTensorSymmetric(T::Matrix{Float64})
    trace = (T[1,1] + T[2,2]) / 2
    cplx = (T[1,1] - T[2,2])/2 + T[1,2]*im
    r = abs(cplx)
    θ = angle(cplx)
    return (trace,r,θ)
end

function recomposeTensorSymmetric(trace,r,θ)
    return [ trace + r*cos(θ) r*sin(θ) ; r*sin(θ) trace - r*cos(θ) ]
end

function classifyTensorEigenvector(yr::AbstractFloat, ys::AbstractFloat)
    if abs(yr) < ϵ
        return SYMMETRIC
    elseif yr > 0
        if abs(yr-ys) < ϵ
            return PI_BY_4
        elseif yr > ys
            return W_CN
        else
            return W_RN
        end
    else
        if abs(yr+ys) < ϵ
            return MINUS_PI_BY_4
        elseif -yr > ys
            return W_CS
        else
            return W_RS
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

function classifyEdgeEigenvalue( t1::FloatMatrix, t2::FloatMatrix )
    decomp1 = decomposeTensor(t1)
    decomp2 = decomposeTensor(t2)
    return classifyEdgeEigenvalue(decomp1..., decomp2...)
end

# returns a list of numbers with the following entries (ordered)
# as one moves from t2 -> t1 (small to large interp values)
# values are -1 signifying r=-d,
#             1 signifying r=d
#             0 signifying anisotropic stretching starts or ends being dominant
# these correspond to vertex patterns from the visualization.
function classifyEdgeEigenvalue( d1::AbstractFloat, r1::AbstractFloat, s1::AbstractFloat, θ1::AbstractFloat, d2::AbstractFloat, r2::AbstractFloat, s2::AbstractFloat, θ2::AbstractFloat )

    y1 = (d2-r2) / ( (d2-r2) - (d1 - r1) )
    y2 = (d2+r2) / ( (d2+r2) - (d1 + r1) )

    if 0 <= y1 <= 1
        if 0 <= y2 <= 1
            if y1 < y2
                cross_values = [(1, y1, 0), (-1,y2,0)]
            else
                cross_values = [(-1,y2, 0), (1,y1,0)]
            end
        else
            cross_values = [(1,y1,0)]
        end
    else
        if 0 <= y1 <= 1
            cross_values = [(-1,y2,0)]
        else
            cross_values = []
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

        # Now store important markers in cross_values as a tuple
            # 1st value: 0 (to distinguish from the other crossing types)
            # 2nd value: interp value in [0,1]
            # 3rd value: 1 for d, 2 for r

        if category != 0 && category != 1
            square_root = sqrt(v^2-4*u*w)
            t1 = (2*w-v + square_root) / (2*u - 2*v + 2*w)
            t2 = (2*w-v - square_root) / (2*u - 2*v + 2*w)

            small_t = min(t1, t2)
            large_t = max(t1, t2)

            if category == 2 || category == 3
                if 0 <= small_t <= 1
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
    sort!(cross_values, by=f(x)=x[2])
    edge = []
    
    # 1st entry - s is larger than d
    # 2nd entry - s is larger than r
    s_is_larger = [ intercept_categories[1] % 2 == 0, intercept_categories[2] % 2 == 0 ]

    for c in cross_values

        if c[1] == 0
            if (c[3] == 1 || s_is_larger[1]) && (c[3] == 2 || s_is_larger[2])
                push!(edge, 0)
            end
            s_is_larger[c[3]] ⊻= true            
        elseif !( s_is_larger[1] && s_is_larger[2] )
            push!(edge, 0)
        end
    end

    return edge

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