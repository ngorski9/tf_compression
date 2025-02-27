include("utils.jl")
include("conicUtils.jl")
include("cellTopology.jl")
include("tensorField.jl")

using ..utils
using ..conicUtils
using ..cellTopology
using ..tensorField

function main()
    tf = loadTensorField2dFromFolder("../output/slice", (101,101,1))
    for i in 1:100
        for j in 1:100
            for k in 0:1
                top = Bool(k)
                topology = classifyCellEigenvalue(getTensorsAtCell(tf,i,j,1,top)..., true)
                topology2 = classifyCellEigenvalue(getTensorsAtCell(tf,i,j,1,top)..., true,true)
                if topology != topology2
                    tensors = getTensorsAtCell(tf,i,j,1,top)
                    println(tensors[1])
                    println(tensors[2])
                    println(tensors[3])
                    println(topology)
                    println(topology2)
                    exit()
                end
            end
        end
    end

end

main()