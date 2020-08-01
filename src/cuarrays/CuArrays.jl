module CuArrays
using CUDA, AutoGrad
using ..KnetArrays: @knet8, @knet8r, reduction_ops
include("autograd.jl")
include("dropout.jl")
include("getindex.jl")
include("reduction.jl")
include("loss.jl")
include("train.jl")
end
