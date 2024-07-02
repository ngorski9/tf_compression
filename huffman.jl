module huffman

export huffmanEncode
export huffmanDecode

abstract type HuffmanNode end

mutable struct HuffmanInnerNode <: HuffmanNode
    left::HuffmanNode
    right::HuffmanNode
    value::Int64
end

mutable struct HuffmanLeafNode <: HuffmanNode
    name::Int64
    value::Int64
end

function getCodeList(node::HuffmanInnerNode)
    codes = Dict()
    left_codes = getCodeList(node.left)
    right_codes = getCodeList(node.right)
    for k in keys(left_codes)
        codes[k] = "0" * left_codes[k]
    end
    for k in keys(right_codes)
        codes[k] = "1" * right_codes[k]
    end
    return codes
end

function getCodeList(node::HuffmanLeafNode)
    return Dict(node.name => "")
end

function getCodeList(frequencies::Dict{Int64, Int64})
    nodes = Array{HuffmanNode}(undef, 0)
    sortedKeys = sort(collect(keys(frequencies)))
    for k in sortedKeys
        push!(nodes, HuffmanLeafNode(k, frequencies[k]))
    end

    while length(nodes) > 1
        sort!(nodes, rev=true, by= f(x)=x.value )
        small_1 = pop!(nodes)
        small_2 = pop!(nodes)
        push!(nodes, HuffmanInnerNode(small_1, small_2, small_1.value+small_2.value))
    end

    return getCodeList(nodes[1])
end

function bitStringToByte(bitString)
    byte::UInt8 = 0
    for j in 1:8
        if bitString[j] == '1'
            byte += 2^(j-1)
        end
    end
    return byte
end

# Codes can only be integer type. Currently, must be int 8 but we'll change that if we need to.
function huffmanEncode(symbols)
    frequencies = Dict{Int64, Int64}()

    for s in symbols
        if haskey(frequencies, s)
            frequencies[s] += 1
        else
            frequencies[s] = 1
        end
    end

    bitString = ""
    output::Array{UInt8} = []
    for s in symbols
        bitString *= codes[s]
        while length(bitString) >= 8

            byte = bitStringToByte(bitString[1:8])

            push!(output, byte)
            bitString = bitString[9:end]

        end
    end

    if mod == 0
        padding = 0
    else
        padding = 8 - length(bitString)
        bitString *= ("0"^padding)
        byte = bitStringToByte(bitString)
        push!(output, byte)
    end

    symbols = collect(keys(frequencies))
    numSymbols = length(symbols)
    header = Array{Int64}(undef, 2*numSymbols+2)
    header[1] = numSymbols
    header[2] = padding
    for i in 1:numSymbols
        header[2*i+1] = symbols[i]
        header[2*i+2] = frequencies[symbols[i]]
    end

    header = reinterpret(UInt8, header)
    return cat(header, output, dims=(1,1))

end

function huffmanDecode(bytes::Array{UInt8})
    tableSize, padding = reinterpret(Int64, bytes[1:16])
    header = reinterpret(Int64, bytes[17:16+16*tableSize])
    body = bytes[17+16*tableSize:length(bytes)]
    frequencies = Dict{Int64, Int64}()
    for i in 1:tableSize
        frequencies[header[2*i-1]] = header[2*i]
    end

    bitString = ""

    codes_by_value = getCodeList(frequencies)
    codes = Dict{String, Int64}()

    for k in keys(codes_by_value)
        codes[codes_by_value[k]] = k
    end

    out::Array{Int64} = []

    for byte in body[1:end-1]
        for i in 1:8
            if byte % 2 == 1
                bitString *= "1"
            else
                bitString *= "0"
            end
            byte ÷= 2
        end            

        code = ""
        i = 1
        while i <= length(bitString)
            code *= bitString[i]
            if haskey(codes, code)
                push!(out, codes[code])
                bitString = bitString[length(code)+1:end]
                i = 1
                code = ""
            else
                i += 1
            end
        end

    end

    byte = body[end]
    for i in 1:8
        if byte % 2 == 1
            bitString *= "1"
        else
            bitString *= "0"
        end
        byte ÷= 2
    end    
    bitString = bitString[1:end-padding]    

    code = ""
    for i in eachindex(bitString)
        code *= bitString[i]
        if haskey(codes, code)
            push!(out, codes[code])
            code = ""
        end
    end

    return out

end

end