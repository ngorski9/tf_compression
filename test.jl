macro test(a)
    b = esc(a)
    return :( println($b) )
end

function main()
    y = 1
    println(y)
end

main()