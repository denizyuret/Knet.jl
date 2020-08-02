module Knet
using AutoGrad, CUDA, Random

include("util.jl")
include("ops/Ops20.jl")
include("train/Train20.jl")
include("data/Data20.jl")
include("knetarrays/KnetArrays.jl")
include("cuarrays/CuArrays.jl")

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

using CUDA; export CuArray
using AutoGrad; export AutoGrad, cat1d, @diff, grad, gradloss, Param, params, value, @gcheck
using .Ops20; export relu, selu, elu, sigm, invx, dropout, bmm, conv4, pool, unpool, deconv4, logp, softmax, logsoftmax, logsumexp, nll, accuracy, zeroone, logistic, bce, mat
using .Data20; export minibatch
using .Train20
# TODO: remove deprecated functions
using .KnetArrays; export KnetArray, RNN, rnninit, rnnforw, rnnparam, rnnparams, batchnorm, bnmoments, bnparams, gpu, gpucopy, cpucopy, knetgc, setseed


# The rest are from Train20: TODO: make this list smaller.
export		# ref:reference.md tut:tutorial
#    accuracy,	# ref, tut
    adadelta!,	# ref
    Adadelta,	# ref
    adadelta,	# ref
    adagrad!,	# ref
    Adagrad,	# ref
    adagrad,	# ref
    adam!,	# ref
    Adam,	# ref
    adam,	# ref, tut
#    AutoGrad,	# ref, tut
#    batchnorm,	# ref
#    bce,	# ref
    bilinear,	# ref
#    bmm,	# ref
#    bnmoments,	# ref
#    bnparams,	# ref
#    cat1d,	# ref
#    conv4,	# ref, tut
    converge!,	# ref
    converge,	# ref, tut
#    cpucopy,	# ref
#    CuArray,
    #Data,	# tut, use Knet.Data
#    deconv4,	# ref
#    @diff,	# ref, tut
    #dir,	# ref, tut, use Knet.dir
#    dropout,	# ref, tut
#    elu,	# ref
    #epochs,	# deprecated, use repeat(data,n)
    gaussian,	# ref
    #gc,  	# ref, tut, use Knet.gc
    #@gheck,	# ref, use AutoGrad.@gcheck
    goldensection, # ref
#    gpu,	# ref, tut
#    gpucopy,	# ref
#    grad,	# ref, tut
#    gradloss,	# ref
    hyperband,	# ref
#    invx,	# ref
#    KnetArray,	# ref, tut
#    knetgc,     # deprecated, use Knet.gc
    #load,	# ref, tut
    #@load,	# ref
#    logistic,	# ref
#    logp,	# ref
#    logsoftmax,	# ref
#    logsumexp,	# ref
#    mat,	# ref, tut
    # minibatch,	# ref, tut
    #minimize!,	# use sgd!, adam! etc.
    #minimize,	# use sgd, adam etc.
    momentum!,	# ref
    Momentum,	# ref
    momentum,	# ref
    nesterov!,	# ref
    Nesterov,	# ref
    nesterov,	# ref
#    nll,	# ref, tut
    optimizers,	# deprecated, use sgd etc.
#    Param,	# ref, tut
    param,	# ref, tut
    param0,	# ref, tut
#    params,	# ref, tut
#    pool,	# ref, tut
    #@primitive, # ref, use AutoGrad.@primitive
    progress!,	# ref, tut
    progress,	# ref, tut
#    relu,	# ref, tut
    rmsprop!,	# ref
    Rmsprop,	# ref
    rmsprop,	# ref
#    RNN,	# ref, tut
#    rnninit,    # deprecated, use RNN
#    rnnparam,	# ref, rnnparam(r,w,l,i,d) deprecated, use rnnparam(r,l,i,d)
#    rnnparams,	# ref, rnnparams(r,w) deprecated, use rnnparams(r)
    #save,	# ref, tut, use Knet.save
    #@save,	# ref, use Knet.@save
    #seed!,	# ref, use Knet.seed!
#    selu,	# ref
#    setseed,	# deprecated, use Knet.seed!
    sgd!,	# ref
    SGD,	# ref
    Sgd,	# deprecated, use SGD
    sgd,	# ref, tut
#    sigm,	# ref
#    softmax,	# ref
    train!,	# deprecated, use sgd, adam etc.
    #train,	# deprecated, use sgd, adam etc.
#    training,	# ref, tut
#    unpool,	# ref
    update!,	# ref
    #updates,	# deprecated, use take(cycle(data),n)
#    value,	# ref, tut
    xavier,	# ref, tut
    xavier_uniform,	# ref
    xavier_normal,	# ref
    #@zerograd, # ref, use AutoGrad.@zerograd
    zeroone	# ref, tut

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
