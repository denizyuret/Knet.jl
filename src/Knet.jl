VERSION >= v"0.4.0-dev+6521" && __precompile__()

module Knet
using AutoGrad

const libknet8 = Libdl.find_library(["libknet8"], [dirname(@__FILE__)])
"""
Knet.dir(path...) constructs a path relative to Knet root.  For example:

Knet.dir("examples","mnist.jl") => "/home/dyuret/.julia/v0.5/Knet/examples/mnist.jl"
"""
dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

export KnetArray, gpu, relu, sigm, invx, logp, logsumexp, conv4, pool, mat, cpu2gpu, gpu2cpu, deconv4, unpool
export grad, gradloss, gradcheck # from AutoGrad

include("gpu.jl")               # gpu support
include("gpuh.jl")
include("kptr.jl")              # KnetPtr
include("karray.jl")            # KnetArray
include("unary.jl")             # unary operators
include("broadcast.jl")         # elementwise broadcasting operations
include("reduction.jl")         # scalar or vector reductions
include("linalg.jl")            # linear algebra functions
include("conv.jl")              # convolution and pooling
include("update.jl")		# update functions
include("distributions.jl")     # distributions for weight init

export gaussian, xavier, bilinear # export distributions
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
