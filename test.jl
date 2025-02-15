include("huffman.jl")

using ..huffman

function main()
    symbols = []

    for i in 1:1000
        push!(symbols, rand(1:30))
    end

    enc = huffmanEncode(symbols)
    dec = huffmanDecode(enc)
    println(dec == symbols)

end

main()