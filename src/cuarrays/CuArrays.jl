module CuArrays
using CUDA, AutoGrad
using ..KnetArrays: @knet8, @knet8r, reduction_ops
include("autograd.jl")
include("getindex.jl")
include("reduction.jl")
include("train.jl")
include("cuarrays.jl")
include("gcnode.jl")
end
