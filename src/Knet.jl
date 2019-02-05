module Knet

using CuArrays

# To see debug output, start julia with `JULIA_DEBUG=Knet julia`
# To perform profiling, set ENV["KNET_TIMER"] to "true" and rebuild Knet. (moved this to gpu.jl)
# The @dbg macro below evaluates `ex` only when debugging. The @debug macro prints stuff as documented in Julia.
macro dbg(ex); :(if Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug,:none,Knet)!==nothing; $(esc(ex)); end); end

export		# ref:reference.md tut:tutorial
    accuracy,	# ref, tut
    adadelta!,	# ref
    Adadelta,	# ref
    adadelta,	# ref
    adagrad!,	# ref
    Adagrad,	# ref
    adagrad,	# ref
    adam!,	# ref
    Adam,	# ref
    adam,	# ref, tut
    AutoGrad,	# ref, tut
    batchnorm,	# ref
    bce,	# ref
    bilinear,	# ref
    bmm,	# ref
    bnmoments,	# ref
    bnparams,	# ref
    cat1d,	# ref
    conv4,	# ref, tut
    converge!,	# ref
    converge,	# ref, tut
    cpucopy,	# ref
    #Data,	# tut, use Knet.Data
    deconv4,	# ref
    @diff,	# ref, tut
    #dir,	# ref, tut, use Knet.dir
    dropout,	# ref, tut
    elu,	# ref
    #epochs,	# deprecated, use repeat(data,n)
    gaussian,	# ref
    #gc,  	# ref, tut, use Knet.gc
    #@gheck,	# ref, use AutoGrad.@gcheck
    goldensection, # ref
    gpu,	# ref, tut
    gpucopy,	# ref
    grad,	# ref, tut
    gradloss,	# ref
    hyperband,	# ref
    invx,	# ref
    KnetArray,	# ref, tut
    knetgc,     # deprecated, use Knet.gc
    #load,	# ref, tut
    #@load,	# ref
    logistic,	# ref
    logp,	# ref
    logsoftmax,	# ref
    logsumexp,	# ref
    mat,	# ref, tut
    minibatch,	# ref, tut
    #minimize!,	# use sgd!, adam! etc.
    #minimize,	# use sgd, adam etc.
    momentum!,	# ref
    Momentum,	# ref
    momentum,	# ref
    nesterov!,	# ref
    Nesterov,	# ref
    nesterov,	# ref
    nll,	# ref, tut
    optimizers,	# deprecated, use sgd etc.
    Param,	# ref, tut
    param,	# ref, tut
    param0,	# ref, tut
    params,	# ref, tut
    pool,	# ref, tut
    #@primitive, # ref, use AutoGrad.@primitive
    progress!,	# ref, tut
    progress,	# ref, tut
    relu,	# ref, tut
    rmsprop!,	# ref
    Rmsprop,	# ref
    rmsprop,	# ref
    RNN,	# ref, tut
    rnninit,    # deprecated, use RNN
    rnnparam,	# ref, rnnparam(r,w,l,i,d) deprecated, use rnnparam(r,l,i,d)
    rnnparams,	# ref, rnnparams(r,w) deprecated, use rnnparams(r)
    #save,	# ref, tut, use Knet.save
    #@save,	# ref, use Knet.@save
    #seed!,	# ref, use Knet.seed!
    selu,	# ref
    setseed,	# deprecated, use Knet.seed!
    sgd!,	# ref
    SGD,	# ref
    Sgd,	# deprecated, use SGD
    sgd,	# ref, tut
    sigm,	# ref
    softmax,	# ref
    train!,	# deprecated, use sgd, adam etc.
    #train,	# deprecated, use sgd, adam etc.
    training,	# ref, tut
    unpool,	# ref
    update!,	# ref
    #updates,	# deprecated, use take(cycle(data),n)
    value,	# ref, tut
    xavier,	# ref, tut
    #@zerograd, # ref, use AutoGrad.@zerograd
    zeroone	# ref, tut

using AutoGrad
include("gpu.jl");              # gpu
include("uva.jl")
include("karray.jl");           # KnetArray
include("gcnode.jl");
include("ops.jl");
include("unary.jl");            # relu, sigm, invx, elu, selu
include("binary.jl");           # elementwise broadcasting operations
include("reduction.jl");        # sum, max, mean, etc.
include("linalg.jl");           # mat # matmul, axpy!, transpose, (i)permutedims
include("bmm.jl");              # bmm # matmul, axpy!, transpose, (i)permutedims
include("conv.jl");             # conv4, pool, deconv4, unpool
include("batchnorm.jl");        # batchnorm, bnmoments, bnparams
include("rnn.jl");              # RNN, rnnparam, rnnparams
include("data.jl");             # Data, minibatch
include("progress.jl");         # progress, progress!
include("train.jl");		# train, train!, minimize, minimize!, converge, converge!, param, param0
include("loss.jl");             # logp, logsoftmax, logsumexp, softmax, nll, logistic, bce, accuracy, zeroone # TODO: PR
include("dropout.jl");          # dropout
include("update.jl"); 		# SGD, Sgd, sgd, sgd!, Momentum, momentum, momentum!, Nesterov, nesterov, nesterov!, Adam, adam, adam!, Adagrad, adagrad, adagrad!, Adadelta, adadelta, adadelta!, Rmsprop, rmsprop, rmsprop!, update!, optimizers
include("distributions.jl"); 	# gaussian, xavier, bilinear
include("random.jl");           # setseed  # TODO: deprecate setseed
include("hyperopt.jl");         # hyperband, goldensection
include("serialize.jl");        # gpucopy,cpucopy
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

#using  AutoGrad: @diff, Param, params, grad, gradloss, value, cat1d, @primitive, @zerograd, @primitive1, @zerograd1, forw, back, Value, AutoGrad
#export AutoGrad, @diff, Param, params, grad, gradloss, value, cat1d #@primitive, @zerograd, @primitive1, @zerograd1, forw, back, Value, getval

end # module
