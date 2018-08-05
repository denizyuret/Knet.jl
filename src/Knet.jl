#__precompile__() rened

module Knet

using Compat, Compat.Libdl, Compat.Random, Compat.Pkg, Compat.LinearAlgebra

IdType = VERSION < v"0.7.0-DEV.3439" ? ObjectIdDict : IdDict

# To see debug output, set DBGFLAGS to non-zero. Each bit of DBGFLAGS
# can be used to show a subset of dbg messages indicated by the `bit`
# argument to the `dbg` macro.
const DBGFLAGS = 0
macro dbg(bit,x); if (1<<bit) & DBGFLAGS != 0; esc(:(println(_dbg($x)))); end; end;

# To perform profiling, set PROFILING to true. (moved this to gpu.jl)
# const PROFILING = false
# macro gs(); if PROFILING; esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))); end; end

const libknet8 = Libdl.find_library(["libknet8"], [joinpath(dirname(@__DIR__),"deps")])

using AutoGrad; export grad, gradloss, gradcheck, getval

include("gpu.jl");              export gpu
include("uva.jl")
include("kptr.jl");             export knetgc # KnetPtr
include("karray.jl");           export KnetArray
include("unfuse.jl");           # julia6 broadcast fixes
include("unary.jl");            export relu, sigm, invx
include("broadcast.jl");        # elementwise broadcasting operations
include("reduction.jl");        # sum, max, mean, etc.
include("linalg.jl");           export mat # matmul, axpy!, transpose, (i)permutedims
include("conv.jl");             export conv4, pool, deconv4, unpool
include("batchnorm.jl");        export batchnorm, bnmoments, bnparams
include("rnn.jl");              export rnnforw, rnninit, rnnparam, rnnparams
include("loss.jl");             export logp, logsumexp, nll, accuracy
include("dropout.jl");          export dropout
include("update.jl"); 		export Sgd, Momentum, Nesterov, Adam, Adagrad, Adadelta, Rmsprop, update!, optimizers
include("distributions.jl"); 	export gaussian, xavier, bilinear
include("random.jl");           export setseed
include("hyperopt.jl");         export hyperband, goldensection
include("data.jl");             export minibatch

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
        # info(r >= 0 ? "Knet using GPU $r" : "No GPU found, Knet using the CPU")
    catch e
        gpu(false)
        # warn("Knet using the CPU: $e")
    end
end

end # module
