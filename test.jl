include("huffman.jl")

using ..huffman

function main()
    freq = Dict{Int64,Int64}(1 => 1000, 2 => 500, 3 => 2, 4 => 2)
    symbols = [1,2,3,4]
    tree = huffman.makeHuffmanTree(freq,symbols)
    codes = huffman.getHuffmanCodes(tree)
    println(codes)
end

main()