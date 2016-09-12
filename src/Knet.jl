VERSION >= v"0.4.0-dev+6521" && __precompile__()

module Knet
using AutoGrad
importall Base
export grad, gradcheck, KnetArray, gpu, gpuinfo, relu, sigm, invx, logp

include("gpu.jl")               # gpu support
include("karray.jl")            # use KnetArrays
include("cuda1.jl")             # unary operators
include("cuda01.jl")            # scalar,array->array
include("cuda10.jl")            # array,scalar->array
include("cuda11.jl")            # array,array->array (elementwise)
include("cuda12.jl")            # array,array->array (broadcasting)
include("cuda20.jl")            # array->scalar (reductions)
include("cuda21.jl")            # array->vector (reductions)
include("cuda22.jl")            # array,array->array (linear algebra)
include("gradcheck.jl")

end # module
