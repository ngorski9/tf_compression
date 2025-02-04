const G = 4

function print_value(x)
    println(x)
end

macro test(var::Symbol, val::Int64)
    return :(
        if $(esc(var)) == $val
            println("hi")
        end
        )
end

function main()
    @test(4,G)
end

main()