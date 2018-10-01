module Knet
using Libdl
# using LinearAlgebra, Statistics, SpecialFunctions, Libdl

# To see debug output, start julia with `JULIA_DEBUG=Knet julia`
# To perform profiling, set ENV["KNET_TIMER"] to "true" and rebuild Knet. (moved this to gpu.jl)
# The @dbg macro below evaluates `ex` only when debugging. The @debug macro prints stuff as documented in Julia.
macro dbg(ex); :(if Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug,:none,Knet)!==nothing; $(esc(ex)); end); end

const libknet8 = Libdl.find_library(["libknet8"], [joinpath(dirname(@__DIR__),"deps")])

using  AutoGrad: @diff, Param, params, grad, gradloss, value, cat1d, @primitive, @zerograd, @primitive1, @zerograd1, forw, back, Value, AutoGrad
export AutoGrad, @diff, Param, params, grad, gradloss, value, cat1d #@primitive, @zerograd, @primitive1, @zerograd1, forw, back, Value, getval

include("gpu.jl");              export gpu
include("uva.jl")
include("kptr.jl");             export knetgc # KnetPtr
include("karray.jl");           export KnetArray
include("gcnode.jl");
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
include("loss.jl");             export logp, logsoftmax, logsumexp, softmax, nll, logistic, bce, accuracy, zeroone # TODO: PR
include("dropout.jl");          export dropout
include("update.jl"); 		export SGD, Sgd, Momentum, Nesterov, Adam, Adagrad, Adadelta, Rmsprop, update!, optimizers
include("distributions.jl"); 	export gaussian, xavier, bilinear
include("random.jl");           export setseed  # TODO: deprecate setseed
include("hyperopt.jl");         export hyperband, goldensection
include("serialize.jl");        export gpucopy,cpucopy
include("jld.jl");              # load, save, @load, @save; not exported use with Knet. prefix.


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
