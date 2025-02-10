const G = 4

function print_value(x)
    println(x)
end

function print_two(x)
    println("2")
end

macro test(var, fun)
    a = print_two
    return :(begin
        b = 3
        println(b)
        ($a)($(esc(var)))
        end)
end

function main()
    @test(4,print_two)
end

main()