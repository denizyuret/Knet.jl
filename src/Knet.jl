module Knet

include("libknet8/LibKnet8.jl")
include("knetarrays/KnetArrays.jl")
include("cuarrays/CuArrays.jl")
include("autograd_gpu/AutoGrad_gpu.jl")
include("ops20/Ops20.jl")
include("ops20_gpu/Ops20_gpu.jl")
include("fileio_gpu/FileIO_gpu.jl")
include("train20/Train20.jl")

# See if we have a gpu at initialization:
import AutoGrad, CUDA
function __init__()
    if CUDA.functional()
        Knet.Train20.array_type[] = Knet.KnetArrays.KnetArray{Float32}
        AutoGrad.set_gc_function(Knet.KnetArrays.cuallocator[] ? Knet.AutoGrad_gpu.gcnode : Knet.AutoGrad_gpu.knetgcnode)
    end
end

"Construct a path relative to Knet root, e.g. Knet.dir(\"examples\") => \"~/.julia/dev/Knet/examples\""
dir(path...)=joinpath(dirname(@__DIR__),path...)

# Match export list with v1.3.9 for backward compatibility
using AutoGrad #: @diff, AutoGrad, Param, cat1d, grad, gradloss, params, value
using Knet.LibKnet8 #: libknet8, @knet8, @knet8r, gpu
using Knet.KnetArrays #: KnetArray, gc, knetgc, ka, setseed, seed!
using Knet.FileIO_gpu #: cpucopy, gpucopy
using Knet.Ops20 #: RNN, accuracy, batchnorm, bce, bmm, bnmoments, bnparams, conv4, deconv4, dropout, elu, invx, logistic, logp, logsoftmax, logsumexp, mat, nll, pool, relu, rnninit, rnnparam, rnnparams, selu, sigm, softmax, unpool, zeroone
using Knet.Train20 #: Adadelta, Adagrad, Adam, Momentum, Nesterov, Rmsprop, SGD, Sgd, adadelta, adadelta!, adagrad, adagrad!, adam, adam!, atype, bilinear, converge, converge!, gaussian, goldensection, hyperband, minibatch, momentum, momentum!, nesterov, nesterov!, optimizers, param, param0, progress, progress!, rmsprop, rmsprop!, sgd, sgd!, train!, training, update!, xavier, xavier_normal, xavier_uniform

export @diff, Adadelta, Adagrad, Adam, AutoGrad, Knet, KnetArray, Momentum, Nesterov, Param, RNN, Rmsprop, SGD, Sgd, accuracy, adadelta, adadelta!, adagrad, adagrad!, adam, adam!, batchnorm, bce, bilinear, bmm, bnmoments, bnparams, cat1d, conv4, converge, converge!, cpucopy, deconv4, dropout, elu, gaussian, goldensection, gpu, gpucopy, grad, gradloss, hyperband, invx, ka, knetgc, logistic, logp, logsoftmax, logsumexp, mat, minibatch, momentum, momentum!, nesterov, nesterov!, nll, optimizers, param, param0, params, pool, progress, progress!, relu, rmsprop, rmsprop!, rnninit, rnnparam, rnnparams, selu, setseed, sgd, sgd!, sigm, softmax, train!, training, unpool, update!, value, xavier, xavier_normal, xavier_uniform, zeroone

end # module


