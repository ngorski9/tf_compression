include("huffman.jl")
using .huffman

function main()
    a::Array{Int8} = [1,2,3,4,4,4,4,3,3,3,3,2,3,3,3,2,1]
    b = huffmanEncode(a, Int8)
    c = huffmanDecode(b, Int8)

    println(length(b))

    for i in eachindex(a)
        if a[i] != c[i]
            println("bad")
            exit()
        end
    end
    println("good")

end

main()