module Knet
importall Base
export KnetArray, tmplike, tmpfree, gpuinfo, relu, sigm
const libknet8 = Libdl.find_library(["libknet8"], [Pkg.dir("Knet/src")])
#include("tmplike0.jl")         # don't use memory management
#include("karray0.jl")          # use CudaArrays
include("tmplike.jl")           # memory management
include("karray.jl")            # use KnetArrays
include("cuda1.jl")             # unary operators
include("cuda01.jl")            # scalar,array->array
include("cuda10.jl")            # array,scalar->array
include("cuda11.jl")            # array,array->array (elementwise)
include("cuda12.jl")            # array,array->array (broadcasting)
include("cuda20.jl")            # array->scalar (reductions)
include("cuda21.jl")            # array->vector (reductions)
include("cuda22.jl")            # array,array->array (linear algebra)
end # module
