module huffman

using DataStructures

export huffmanEncode
export huffmanDecode

# a huffman tree will simply be an array of these
struct HuffmanNode
    elt::Int64
    left::Int64
    right::Int64
end

# hopefully we will not use these any more...
# function getCodeList(node::HuffmanInnerNode)
#     codes = Dict{Int64, String}()
#     left_codes = getCodeList(node.left)
#     right_codes = getCodeList(node.right)
#     for k in keys(left_codes)
#         codes[k] = "0" * left_codes[k]
#     end
#     for k in keys(right_codes)
#         codes[k] = "1" * right_codes[k]
#     end
#     return codes
# end

# function getCodeList(node::HuffmanLeafNode)
#     return Dict(node.name => "")
# end

# function getCodeList(frequencies::Dict{Int64, Int64})
#     nodes::Array{HuffmanNode} = Array{HuffmanNode}(undef, 0)
#     sortedKeys = sort(collect(keys(frequencies)))
#     for k in sortedKeys
#         push!(nodes, HuffmanLeafNode(k, frequencies[k]))
#     end

#     while length(nodes) > 1
#         sort!(nodes, rev=true, by= f(x)=x.value )
#         small_1 = pop!(nodes)
#         small_2 = pop!(nodes)
#         push!(nodes, HuffmanInnerNode(small_1, small_2, small_1.value+small_2.value))
#     end

#     return getCodeList(nodes[1])
# end

# function bitStringToByte(bitString)
#     byte::UInt8 = 0
#     for j in 1:8
#         if bitString[j] == '1'
#             byte += 2^(j-1)
#         end
#     end
#     return byte
# end

function makeHuffmanTree(frequencies::Dict{Int64,Int64}, symbols::Array{Int64})
    numSymbols = length(symbols)
    heapList::Array{Tuple{Int64,Int64,Int64}} = Array{Tuple{Int64,Int64,Int64}}(undef,numSymbols)

    for i in 1:numSymbols
        heapList[i] = (symbols[i],-1,frequencies[symbols[i]])
    end

    heap = BinaryHeap{Tuple{Int64,Int64,Int64}}(Base.By(last),heapList)
    
    numNodes = 2*numSymbols-1
    tree = Array{HuffmanNode}(undef,numNodes)

    iter = 1
    while iter <= numNodes
        small1 = pop!(heap)
        small2 = pop!(heap)
        if small1[2] == -1
            tree[iter] = HuffmanNode(small1[1],-1,-1)
            small1Index = iter
            iter += 1
        else
            small1Index = small1[2]
        end

        if small2[2] == -1
            tree[iter] = HuffmanNode(small2[1],-1,-1)
            small2Index = iter
            iter += 1
        else
            small2Index = small2[2]
        end

        tree[iter] = HuffmanNode(-1,small1Index,small2Index)
        push!(heap, (-1,iter,small1[3]+small2[3]))
        iter += 1
    end

    return tree
end

function getHuffmanCodes(tree::Array{HuffmanNode})
    node = length(tree)
    codeSoFar::Array{Int8} = []
    codes = Dict{Int64,Array{Int8}}()
    recursiveGetHuffmanCodes(tree, node, codes, codeSoFar)
    return codes
end

function recursiveGetHuffmanCodes(tree::Array{HuffmanNode}, node::Int64, codes::Dict{Int64,Array{Int8}}, codeSoFar::Array{Int8})
    if tree[node].left == -1 # eg if this is a leaf node
        codes[tree[node].elt] = copy(codeSoFar)
    else
        push!(codeSoFar,0)
        recursiveGetHuffmanCodes(tree, tree[node].left, codes, codeSoFar)
        pop!(codeSoFar)
        push!(codeSoFar,1)
        recursiveGetHuffmanCodes(tree, tree[node].right, codes, codeSoFar)
        pop!(codeSoFar)
    end
end

# # Codes can only be integer type. Currently, must be Int64 but we'll change that if we need to.
# function huffmanEncode(symbols)
#     frequencies = Dict{Int64, Int64}()
#     symbols::Array{Int64} = Array{Int64}([])

#     for s in symbols
#         sI64 = Int64(s)
#         if haskey(frequencies, sI64)
#             frequencies[sI64] += 1
#         else
#             frequencies[sI64] = 1
#             push!(symbols, sI64)
#         end
#     end

#     output::Array{UInt8} = []
#     nextOut::UInt8 = 0
#     for s in symbols
#         bitString *= codes[s]
#         while length(bitString) >= 8

#             byte = bitStringToByte(bitString[1:8])

#             push!(output, byte)
#             bitString = bitString[9:end]

#         end
#     end

#     if mod == 0
#         padding = 0
#     else
#         padding = 8 - length(bitString)
#         bitString *= ("0"^padding)
#         byte = bitStringToByte(bitString)
#         push!(output, byte)
#     end

#     symbols = collect(keys(frequencies))
#     numSymbols = length(symbols)
#     header::Array{Int64} = Array{Int64}(undef, 2*numSymbols+2)
#     header[1] = numSymbols
#     header[2] = padding
#     for i in 1:numSymbols
#         header[2*i+1] = symbols[i]
#         header[2*i+2] = frequencies[symbols[i]]
#     end

#     headerBytes::Array{UInt8} = reinterpret(UInt8, header)
#     return cat(headerBytes, output, dims=(1,1))

# end

# function huffmanDecode(bytes::Array{UInt8})
#     tableSize, padding = reinterpret(Int64, bytes[1:16])
#     header = reinterpret(Int64, bytes[17:16+16*tableSize])
#     body = bytes[17+16*tableSize:length(bytes)]
#     frequencies = Dict{Int64, Int64}()
#     for i in 1:tableSize
#         frequencies[header[2*i-1]] = header[2*i]
#     end

#     bitString = ""

#     codes_by_value = getCodeList(frequencies)
#     codes = Dict{String, Int64}()

#     for k in keys(codes_by_value)
#         codes[codes_by_value[k]] = k
#     end

#     out::Array{Int64} = []

#     for byte in body[1:end-1]
#         for i in 1:8
#             if byte % 2 == 1
#                 bitString *= "1"
#             else
#                 bitString *= "0"
#             end
#             byte ÷= 2
#         end            

#         code = ""
#         i = 1
#         while i <= length(bitString)
#             code *= bitString[i]
#             if haskey(codes, code)
#                 push!(out, codes[code])
#                 bitString = bitString[length(code)+1:end]
#                 i = 1
#                 code = ""
#             else
#                 i += 1
#             end
#         end

#     end

#     byte = body[end]
#     for i in 1:8
#         if byte % 2 == 1
#             bitString *= "1"
#         else
#             bitString *= "0"
#         end
#         byte ÷= 2
#     end    
#     bitString = bitString[1:end-padding]    

#     code = ""
#     for i in eachindex(bitString)
#         code *= bitString[i]
#         if haskey(codes, code)
#             push!(out, codes[code])
#             code = ""
#         end
#     end

#     return out

# end

end