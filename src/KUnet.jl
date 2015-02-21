module KUnet

type Layer w; b; fx; fy; dw; db; pw; pb; y; x; dx; dropout; xdrop; Layer()=new() end
typealias Net Array{Layer,1}
type UpdateParam 
    learningRate
    l1reg
    l2reg
    maxnorm
    adagrad;  ada
    momentum; mom
    nesterov; nes
    UpdateParam()=new(0.01)
end

include("layer.jl")
include("net.jl")
include("cuda.jl")
include("update.jl")
include("func.jl")
include("h5io.jl")

end # module
