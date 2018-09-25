module Knet
using Libdl
# using LinearAlgebra, Statistics, SpecialFunctions, Libdl

# To see debug output, start julia with `JULIA_DEBUG=Knet julia`
macro dbg(ex); :(if Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug,:none,Knet)!==nothing; $(esc(ex)); end); end

# To perform profiling, set PROFILING to true. (moved this to gpu.jl)
# const PROFILING = false
# macro gs(); if PROFILING; esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))); end; end

const libknet8 = Libdl.find_library(["libknet8"], [joinpath(dirname(@__DIR__),"deps")])

using AutoGrad # Param, params, grad, value, @diff, gradloss, getval, @primitive, @zerograd, @primitive1, @zerograd1, cat1d
using AutoGrad: forw, back, Value
export grad, gradloss, value, Param, @diff

include("gpu.jl");              export gpu
include("uva.jl")
include("kptr.jl");             export knetgc # KnetPtr
include("karray.jl");           export KnetArray
include("ops.jl");
include("unary.jl");            export relu, sigm, invx, elu, selu
include("broadcast.jl");        # elementwise broadcasting operations
include("reduction.jl");        # sum, max, mean, etc.
include("linalg.jl");           export mat # matmul, axpy!, transpose, (i)permutedims
include("conv.jl");             export conv4, pool, deconv4, unpool
include("batchnorm.jl");        export batchnorm, bnmoments, bnparams
include("rnn.jl");              export rnnforw, rnninit, rnnparam, rnnparams, RNN # TODO: deprecate old interface
include("data.jl");             export Data, minibatch
include("model.jl");		export param, param0, train!, Train
include("loss.jl");             export logp, logsumexp, nll, accuracy, zeroone # TODO: PR
include("dropout.jl");          export dropout
include("update.jl"); 		export SGD, Sgd, Momentum, Nesterov, Adam, Adagrad, Adadelta, Rmsprop, update!, optimizers
include("distributions.jl"); 	export gaussian, xavier, bilinear
include("random.jl");           export setseed  # TODO: deprecate setseed
include("hyperopt.jl");         export hyperband, goldensection
include("jld.jl");              export RnnJLD,KnetJLD
#include("cudnn.jl");		export cudnnsigm, cudnnrelu, cudnntanh, cudnncrelu, cudnnelu, cudnnidentity

"""
    Knet.dir(path...)

Construct a path relative to Knet root.

# Example
```julia
julia> Knet.dir("examples","mnist.jl")
"/home/dyuret/.julia/v0.5/Knet/examples/mnist.jl"
```
"""
dir(path...) = joinpath(dirname(@__DIR__),path...)


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

# @use X,Y,Z calls using on packages installing them if necessary. (WIP)
# 1. still need "using Knet"
# 2. Pkg.insalled gives false for stdlib packages.
# macro use(ps)
#     if isa(ps, Symbol); ps = Expr(:tuple,ps); end
#     a = map(ps.args) do p
#         s=string(p) 
#         esc(:(haskey(Pkg.installed(),$s)||Pkg.add($s); using $p))
#     end
#     Expr(:block,:(using Pkg),a...)
# end
# export @use

end # module
