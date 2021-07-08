using Test, Knet, CUDA
using Knet.Layers20
CUDA.functional() ?  Knet.Layers20.settype!(KnetArray{Float64}) : Knet.Layers20.settype!(Array{Float64})
println("Testing with $(Knet.Layers20.arrtype)")
