module utils

using StaticArrays

export getCellVertexCoords
export adjustAngle
export saveArray32
export saveArray64
export loadArray
export removeIfExists
export remove
export getCodeValue
export pushIfAbsent!
export isClose
export isRelativelyClose
export isGreater
export isRelativelyGreater
export isLess
export isRelativelyLess
export rankOrder

# Numerical constants

export ϵ
export MAX_PRECISION
export compressor
const ϵ = 1e-10
const MAX_PRECISION = 10

export smoothness
const smoothness = 0

# cell types

export CP_NORMAL
export CP_TRISECTOR
export CP_WEDGE
export CP_ZERO_CORNER
export CP_ZERO_EDGE
export CP_ZERO_FULL
export CP_OTHER
export CP_BORDER
export CP_LINE
export CP_FULL
const CP_NORMAL = 0
const CP_TRISECTOR = 1
const CP_WEDGE = 2
const CP_ZERO_CORNER = 3
const CP_ZERO_EDGE = 4
const CP_ZERO_FULL = 5
const CP_OTHER = 6

# used for the full classification that we put in the output
const CP_BORDER = 3
const CP_LINE = 4
const CP_FULL = 5

# θ2 - θ1 > π → trisector
# θ2 - θ1 < π → wedge

# adjustment types

export CODE_NO_CHANGE
export CODE_CHANGE_EIGENVECTOR
export CODE_CHANGE_EIGENVALUE
export CODE_LOSSLESS_D
export CODE_LOSSLESS_R
export CODE_LOSSLESS_S
export CODE_LOSSLESS_ANGLE
export CODE_LOSSLESS_FULL
export CODE_LOSSLESS_FULL_64
const CODE_NO_CHANGE = 1
const CODE_CHANGE_EIGENVECTOR = 2 # we take this code to the power of the class that we are going for.
const CODE_CHANGE_EIGENVALUE = 3 # similarly, take this to the power of the class we are going for.
const CODE_LOSSLESS_D = 5
const CODE_LOSSLESS_R = 7
const CODE_LOSSLESS_S = 11
const CODE_LOSSLESS_ANGLE = 13
const CODE_LOSSLESS_FULL = 17
const CODE_LOSSLESS_FULL_64 = 19 # 💀

export FloatMatrix
export FloatMatrixSymmetric
const FloatMatrix = SMatrix{2,2,Float64,4}
const FloatMatrixSymmetric = SVector{3,Float64}

# eigenvector graph numbering system:
export W_CN
export PI_BY_4
export W_RN
export SYMMETRIC
export W_RS
export MINUS_PI_BY_4
export W_CS
const W_CN = 1
const PI_BY_4 = 2
const W_RN = 3
const SYMMETRIC = 4
const W_RS = 5
const MINUS_PI_BY_4 = 6
const W_CS = 7

# Eigenvector graph colors
export darkGreen
export lightGreen
export lightRed
export darkRed
export eigenvectorColors
const darkGreen = "#00701c"
const lightGreen = "#00ff40"
const lightRed = "#ff3636"
const darkRed = "#610000"
const eigenvectorColors = [darkRed, lightGreen, lightRed, "blue", lightGreen, lightRed, darkGreen]

# Eigenvalue graph region labels and coloring
export POSITIVE_SCALING
export COUNTERCLOCKWISE_ROTATION
export NEGATIVE_SCALING
export CLOCKWISE_ROTATION
export ANISOTROPIC_STRETCHING
export eigenvalueColors
const POSITIVE_SCALING = 1
const COUNTERCLOCKWISE_ROTATION = 2
const NEGATIVE_SCALING = 3
const CLOCKWISE_ROTATION = 4
const ANISOTROPIC_STRETCHING = 5
const eigenvalueColors = ["yellow", "red", "blue", "green", "white"]

export eigenvalueRegionBorders
const eigenvalueRegionBorders = Dict(
    -1 => Dict( POSITIVE_SCALING => CLOCKWISE_ROTATION, COUNTERCLOCKWISE_ROTATION => NEGATIVE_SCALING, NEGATIVE_SCALING => COUNTERCLOCKWISE_ROTATION, CLOCKWISE_ROTATION => POSITIVE_SCALING ),
    1 => Dict( POSITIVE_SCALING => COUNTERCLOCKWISE_ROTATION, COUNTERCLOCKWISE_ROTATION => POSITIVE_SCALING, NEGATIVE_SCALING => CLOCKWISE_ROTATION, CLOCKWISE_ROTATION => NEGATIVE_SCALING )
)

# Grid cells look like this

# o---o
# |\  |        up is y direction.
# | \ |        right is x direction.
# |  \|        the "top" cell has the top border.
# o---o        the "bottom cell has the bottom border.

# Gives out ranks with ties broken in parameter order.
function rankOrder(a,b,c)
    if isGreater(a,b)
        if isGreater(c,a)
            return (2,3,1)
        elseif isClose(c,a)
            return (1,3,2)
        else
            if isGreater(b,c)
                return (1,2,3)
            else
                return (1,3,2)
            end
        end
    elseif isClose(a,b)
        if isGreater(c,a)
            return (2,3,1)
        else
            return (1,2,3)
        end
    else
        if isGreater(c,a)
            if !isLess(b,c)
                return (3,1,2)
            else
                return (3,2,1)
            end
        else
            return (2,1,3)
        end
    end
end

function getCodeValue(code::Int64, key::Int64)
    value = 0
    while code % key == 0
        code ÷= key
        value += 1
    end
    return code, value
end

# Used for the previous check stack in compress
function pushIfAbsent!(stack::Array{Tuple{Int64, Int64, Bool}}, elt::Tuple{Int64, Int64, Bool})
    if elt ∉ stack
        push!(stack, elt)
    end
end

# Returns them in clockwise orientation
function getCellVertexCoords(x::Int64, y::Int64, t::Int64, top::Bool)
    if top
        return [(x,y+1,t), (x+1,y,t), (x+1,y+1,t)]
    else
        return [(x,y,t), (x+1,y,t), (x,y+1,t)]
    end
end

function adjustAngle(angle::Union{Float64, Float32})
    return angle - 2pi*floor(angle/2pi)
end

function saveArray32(filename, array::Array{Float32})
    out_file = open(filename, "w")
    write(out_file, vec(array))
    close(out_file)
end

function saveArray32(filename, array::Array{Float64})
    out_file = open(filename, "w")
    write(out_file, vec(Array{Float32}(array)))
    close(out_file)
end

function saveArray64(filename, array::Array{Float64})
    out_file = open(filename, "w")
    write(out_file, vec(array))
    close(out_file)
end

function loadArray(filename, type, dims=nothing)
    in_file = open(filename, "r")
    array = reinterpret(type, read(in_file))
    close(in_file)

    if !isnothing(dims)
        array = reshape(array, dims)
    end

    return array
end

function removeIfExists(filename)
    try
        run(`rm $filename`)
    catch
    end
end

function remove(filename)
    run(`rm $filename`)
end

function isClose(x1::Float64, x2::Float64)
    return abs(x1-x2) < ϵ
end

function isGreater(x1::Float64, x2::Float64)
    return x1 > x2 + ϵ
end

function isLess(x1::Float64, x2::Float64)
    return x1 < x2 - ϵ
end

function isRelativelyClose(x1::Float64, x2::Float64)
    return abs(x1-x2) < ϵ * max(x1,x2)
end

function isRelativelyGreater(x1::Float64, x2::Float64)
    return x1 > x2 + ϵ * max(x1,x2)
end

function isRelativelyLess(x1::Float64, x2::Float64)
    return x1 < x2 - ϵ * max(x1,x2)
end

end