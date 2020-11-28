module CuArrays

import Base: sum, prod, minimum, maximum, getindex, convert
using CUDA: CuArray, CuPtr
using Knet.LibKnet8: @knet8, @knet8r, reduction_ops
using Knet.KnetArrays: checkbetween

include("convert.jl")
include("getindex.jl")
include("reduction.jl")
include("cubytes.jl"); export cuarrays, cubytes
include("jld2.jl")

end
