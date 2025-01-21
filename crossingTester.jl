include("utils.jl")
include("tensorField.jl")

using .utils
using .tensorField

using Printf

function main()

    folder = "../output/reconstructed"
    dims = (150, 450, 1)
    
    # remember this is 1 indexed so it is off from what we previously did by 1!
    xLow = 76
    xHigh = 79
    yLow = 18
    yHigh = 21
    t = 1
    numOnly = false
    minCrossing = 0.01

    tf = loadTensorField2dFromFolder(folder, dims)

    for y = yLow:yHigh
        for x in xLow:xHigh

            if numOnly
                print("x")
            else
                print("xxxxxxxxxxx")
            end
            if x != xHigh
                t1 = getTensor(tf, x, y, t)
                t2 = getTensor(tf, x+1, y, t)
                evalClass, evalLoc, evecClass, evecLoc = classifyEdgeOuter( t1, t2, minCrossing )

                if numOnly
                    print(" $evecClass ")
                else
                    print(" ")
                    if length(evecLoc) == 0
                        print("///////////")
                    else
                        print(@sprintf("%.5E", evecLoc[1]))
                    end
                    print(" ")
                end
            end

        end

        print("\n")

        # row with diagonals if applicable
        if y != yHigh

            for x in xLow:xHigh

                t1 = getTensor(tf, x, y, t)
                t2 = getTensor(tf, x, y+1, t)
                evalClass, evalLoc, evecClass, evecLoc = classifyEdgeOuter( t1, t2, minCrossing )
                if numOnly
                    print(evecClass)
                else
                    if length(evecLoc) == 0
                        print("///////////")
                    else
                        print(@sprintf("%.5E", evecLoc[1]))
                    end
                end

                if x != xHigh
                    t1 = getTensor(tf, x+1, y, t)
                    t2 = getTensor(tf, x, y+1, t)

                    evalClass, evalLoc, evecClass, evecLoc = classifyEdgeOuter( t1, t2, minCrossing )
                    if numOnly
                        print(" $evecClass ")
                    else
                        print(" ")
                        if length(evecLoc) == 0
                            print("///////////")
                        else
                            print(@sprintf("%.5E", evecLoc[1]))
                        end
                        print(" ")
                    end
                end

            end

            print("\n")

        end
    end

    println("hi")
end

main()