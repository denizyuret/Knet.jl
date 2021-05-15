module Ops21_gpu

using Knet.KnetArrays: KnetArray
using CUDA: CuArray
using AutoGrad: Value
const GPUVal{T,N} = Union{KnetArray{T,N},CuArray{T,N},Value{KnetArray{T,N}},Value{CuArray{T,N}}}

include("activation.jl")
include("linear.jl")
include("batchnorm.jl")
include("conv.jl")
include("pool.jl")

end
