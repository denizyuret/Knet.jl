VERSION >= v"0.4.0-dev+6521" && __precompile__()

module Knet

const libknet8 = Libdl.find_library(["libknet8"], [dirname(@__FILE__)])

using AutoGrad; export grad, gradloss, gradcheck

include("gpu.jl");              export gpu
include("kptr.jl");             # KnetPtr
include("karray.jl");           export KnetArray, cpu2gpu, gpu2cpu
include("unary.jl");            export relu, sigm, invx, logp
include("broadcast.jl");        # elementwise broadcasting operations
include("reduction.jl");        export logsumexp
include("linalg.jl");           export mat # matmul, axpy!, transpose, (i)permutedims
include("conv.jl");             export conv4, pool, deconv4, unpool
include("update.jl"); 		export Sgd, Momentum, Adam, Adagrad, Adadelta, Rmsprop, update!
include("distributions.jl"); 	export gaussian, xavier, bilinear


"""
    Knet.dir(path...)

Construct a path relative to Knet root.

# Example
```julia
julia> Knet.dir("examples","mnist.jl")
"/home/dyuret/.julia/v0.5/Knet/examples/mnist.jl"
```
"""
dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)


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
