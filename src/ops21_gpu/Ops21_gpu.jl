module Ops21_gpu

using Knet.KnetArrays: KnetArray
using CUDA: CuArray
const GPUVal{T,N} = Union{KnetArray{T,N},CuArray{T,N},Value{KnetArray{T,N}},Value{CuArray{T,N}}}

include("activation.jl")
include("mmul.jl")
include("batchnorm.jl")
include("conv.jl")

end
