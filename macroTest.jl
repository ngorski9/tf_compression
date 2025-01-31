const G = 4

function print_value(x)
    println(x)
end

macro test(var)
    return :(@test2($(esc(var)), true))
end

macro test2(var, bool)
    if bool
        return :($(esc(var)) = G)
    else
        return :($(esc(var)) = 99)
    end
end

struct double
    a::Float64
    b::Float64
end

function Base.isless(first::double, second::double)
    if first.a != second.a
        return first.a < second.a
    else
        return first.b < second.b
    end
end

function main()
    first = double(1,4)
    second = double(2,3)
    a = [double(1,4), double(1,5), double(2,3), double(2,4)]
    sort(a)
    println(a)
end

main()