module Knet
importall Base
export KnetArray, tmplike, tmpfree, relu, sigm
const libknet8 = Libdl.find_library(["libknet8"], [Pkg.dir("Knet/src")])
include("karray.jl")            # memory management
include("cuda1.jl")             # unary operators
include("cuda01.jl")            # scalar,array->array
include("cuda10.jl")            # array,scalar->array
include("cuda11.jl")            # array,array->array (elementwise)
include("cuda22.jl")            # array,array->array (linear algebra)
include("cuda12.jl")            # array,array->array (broadcasting)
include("cuda20.jl")            # array->scalar (reductions)
end # module
