VERSION >= v"0.4.0-dev+6521" && __precompile__()

module Knet
using AutoGrad

const libknet8 = Libdl.find_library(["libknet8"], [dirname(@__FILE__)])
"""
Knet.dir(path...) constructs a path relative to Knet root.  For example:

Knet.dir("examples","mnist.jl") => "/home/dyuret/.julia/v0.5/Knet/examples/mnist.jl"
"""
dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

export grad, KnetArray, gradcheck, gpu, relu, sigm, invx, logp, logsumexp, conv4, pool, mat
include("gpu.jl")               # gpu support
include("kptr.jl")              # KnetPtr
include("karray.jl")            # KnetArray
include("cuda1.jl")             # unary operators
include("cuda01.jl")            # scalar,array->array
include("cuda10.jl")            # array,scalar->array
include("cuda11.jl")            # array,array->array (elementwise, same shape)
include("cuda12.jl")            # array,array->array (broadcasting)
include("cuda20.jl")            # array->scalar (reductions)
include("cuda21.jl")            # array->vector (reductions)
include("cuda22.jl")            # array,array->array (linear algebra)
include("cuda44.jl")            # convolution and pooling
include("gradcheck.jl")         # gradient check
include("update.jl")		# update functions

export Sgd, Momentum, Adam, Adagrad, Adadelta, Rmsprop, update!

# See if we have a gpu at initialization:
function __init__()
    try
        r = gpu(true)
        info(r >= 0 ? "Knet using GPU $r" : "No GPU found, Knet using the CPU")
    catch e
        warn("Knet using the CPU: $e")
        gpu(false)
    end
end

end # module
