module Knet

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
    xavier_uniform,	# ref
    xavier_normal,	# ref
    #@zerograd, # ref, use AutoGrad.@zerograd
    zeroone	# ref, tut

using AutoGrad

# using Pkg; const AUTOGRAD_VERSION = (isdefined(Pkg.API,:__installed) ? Pkg.API.__installed()["AutoGrad"] : Pkg.dependencies()[Base.UUID("6710c13c-97f1-543f-91c5-74e8f7d95b35")].version)

include("cuda/ops.jl");
include("knetarrays/gpu.jl");              # gpu
include("knetarrays/kptr.jl");
include("knetarrays/karray.jl");           # KnetArray
include("knetarrays/cuarray.jl");
include("cuarrays/autograd.jl");
include("cuarrays/getindex.jl");
include("knetarrays/gcnode.jl");
include("knetarrays/unary.jl");            # relu, sigm, invx, elu, selu
include("knetarrays/binary.jl");           # elementwise broadcasting operations
include("knetarrays/reduction.jl");        # sum, max, etc.
include("cuarrays/reduction.jl");
include("knetarrays/statistics.jl");       # mean, std, var, stdm, varm
include("knetarrays/linalg.jl");           # mat # matmul, axpy!, transpose, (i)permutedims
include("ops/bmm.jl");              # bmm # matmul, axpy!, transpose, (i)permutedims
include("ops/conv.jl");             # conv4, pool, deconv4, unpool
include("ops/batchnorm.jl");        # batchnorm, bnmoments, bnparams
include("ops/rnn.jl");              # RNN, rnnparam, rnnparams
include("data/data.jl");             # Data, minibatch
include("train/progress.jl");         # progress, progress!
include("train/train.jl");		# train, train!, minimize, minimize!, converge, converge!, param, param0
include("ops/loss.jl");             # logp, logsoftmax, logsumexp, softmax, nll, logistic, bce, accuracy, zeroone # TODO: PR
include("ops/dropout.jl");          # dropout
include("cuarrays/dropout.jl");
include("train/update.jl"); 		# SGD, Sgd, sgd, sgd!, Momentum, momentum, momentum!, Nesterov, nesterov, nesterov!, Adam, adam, adam!, Adagrad, adagrad, adagrad!, Adadelta, adadelta, adadelta!, Rmsprop, rmsprop, rmsprop!, update!, optimizers
include("train/distributions.jl"); 	# gaussian, xavier, bilinear, xavier_uniform, xavier_normal
include("train/hyperopt.jl");         # hyperband, goldensection
include("knetarrays/serialize.jl");        # gpucopy,cpucopy
include("knetarrays/jld.jl");              # load, save, @load, @save; not exported use with Knet. prefix.


"""
    Knet.dir(path...)

Construct a path relative to Knet root.

# Example
```julia
julia> Knet.dir("examples","mnist.jl")
"/home/dyuret/.julia/dev/Knet/examples/mnist.jl"
```
"""
dir(path...) = joinpath(dirname(@__DIR__),path...)


# See if we have a gpu at initialization:
function __init__()
    if !CUDA.functional()
        @warn "Knet cannot use the GPU: CUDA.jl is not functional"
    else; try
        dev = gpu(true)
        if dev >= 0
            AutoGrad.set_gc_function(Knet.knetgcnode)
            @debug "Knet using GPU $dev"
        else
            @debug "No GPU found, Knet using the CPU"
        end
    catch e
        gpu(false)
        @warn "Knet cannot use the GPU: $e"
    end; end
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
