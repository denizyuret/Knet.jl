VERSION >= v"0.4.0-dev+6521" && __precompile__()

module Knet
importall Base
export KnetArray, KA, gpuinfo, knetgc, relu, sigm
const libknet8 = Libdl.find_library(["libknet8"], [Pkg.dir("Knet/src")])

# include("karray0.jl")          # use CudaArrays
include("karray.jl")            # use KnetArrays
include("cuda1.jl")             # unary operators
include("cuda01.jl")            # scalar,array->array
include("cuda10.jl")            # array,scalar->array
include("cuda11.jl")            # array,array->array (elementwise)
include("cuda12.jl")            # array,array->array (broadcasting)
include("cuda20.jl")            # array->scalar (reductions)
include("cuda21.jl")            # array->vector (reductions)
include("cuda22.jl")            # array,array->array (linear algebra)

function __init__()
    handleP = Ptr{Void}[0]
    cublascheck(ccall((:cublasCreate_v2, libcublas), UInt32, (Ptr{Ptr{Void}},), handleP))
    global cublashandle = handleP[1]
    atexit(()->cublascheck(ccall((:cublasDestroy_v2, libcublas), UInt32, (Ptr{Void},), cublashandle)))
end    

end # module
