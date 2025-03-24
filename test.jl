using StaticArrays

function cyclicMatch(M1::MVector{10,Int8}, M2::MVector{10,Int8})
    l = 10
    for i in 1:10
        if M1[i] == 0
            if M2[i] != 0
                return false
            else
                l = i-1
                break
            end
        end
    end

    if l == 0
        return true
    end

    for off in 0:l-1
        match = true
        for i in 1:l
            if M1[i] != M2[ (i-1 + off) % l + 1 ]
                match = false
                break
            end
        end
        if match
            return true
        end
    end

    return false
end

function main()
    M1 = MArray{Tuple{10},Int8}(1,2,3,4,5,0,0,0,0,0)
    M2 = MArray{Tuple{10},Int8}(3,4,5,1,2,0,0,0,0,0)
    println(cyclicMatch(M1,M2))

end

main()