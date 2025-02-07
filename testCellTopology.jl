include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")
include("tensorField.jl")

using ..utils
using ..conicUtils
using ..cellTopology
using ..tensorField

function main()
    tf = loadTensorField2dFromFolder("../output", (101,101,1))
    outf = open("../new.txt", "w")
    for i in 1:100
        for j in 1:100
            for k in 0:1
                top = Bool(k)
                topology = classifyCellEigenvalue(getTensorsAtCell(tf,i,j,1,top)..., true)
                write(outf, string(topology) * "\n")
            end
        end
    end

    close(outf)
end

main()