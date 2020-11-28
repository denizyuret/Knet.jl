module Knet

"Construct a path relative to Knet root, e.g. Knet.dir(\"examples\") => \"~/.julia/dev/Knet/examples\""
dir(path...)=joinpath(dirname(@__DIR__),path...)

"Default array and element type used by Knet, override by setting Knet.atype() or Knet.array_type[]"
atype() = array_type[]
atype(x) = convert(atype(),x)
const array_type = Ref{Type}(Array{Float32})

include("libknet8/LibKnet8.jl")
include("knetarrays/KnetArrays.jl")
include("cuarrays/CuArrays.jl")
include("autograd_gpu/AutoGrad_gpu.jl")
include("ops20/Ops20.jl")
include("ops20_gpu/Ops20_gpu.jl")
include("ops21/Ops21.jl")
include("ops21_gpu/Ops21_gpu.jl")
include("train20/Train20.jl")
# include("layers21/Layers21.jl")

# See if we have a gpu at initialization:
import AutoGrad, CUDA
function __init__()
    if CUDA.functional()
        if isempty(Knet.LibKnet8.libknet8)
            @warn "libknet8 library not found, some GPU functionality may not be available, try reinstalling Knet."
        end
        Knet.array_type[] = Knet.KnetArrays.KnetArray{Float32}
        AutoGrad.set_gc_function(Knet.KnetArrays.cuallocator[] ? Knet.AutoGrad_gpu.gcnode : Knet.AutoGrad_gpu.knetgcnode)
        if CUDA.has_nvml() # Pick the device with highest memory
            mem(d) = CUDA.NVML.memory_info(CUDA.NVML.Device(CUDA.uuid(d))).free
            CUDA.device!(argmax(Dict(d=>mem(d) for d in CUDA.devices())))
        end
    end
end

# Match export list with v1.3.9 for backward compatibility
using AutoGrad #: @diff, AutoGrad, Param, cat1d, grad, gradloss, params, value
using Knet.LibKnet8 #: libknet8, @knet8, @knet8r, gpu
using Knet.KnetArrays #: KnetArray, gc, knetgc, ka, setseed, seed!
using Knet.Ops20 #: RNN, accuracy, batchnorm, bce, bmm, bnmoments, bnparams, conv4, deconv4, dropout, elu, invx, logistic, logp, logsoftmax, logsumexp, mat, nll, pool, relu, rnnforw, rnninit, rnnparam, rnnparams, selu, sigm, softmax, unpool, zeroone
using Knet.Train20 #: Adadelta, Adagrad, Adam, Momentum, Nesterov, Rmsprop, SGD, Sgd, adadelta, adadelta!, adagrad, adagrad!, adam, adam!, atype, bilinear, converge, converge!, gaussian, goldensection, hyperband, minibatch, momentum, momentum!, nesterov, nesterov!, optimizers, param, param0, progress, progress!, rmsprop, rmsprop!, sgd, sgd!, train!, training, update!, xavier, xavier_normal, xavier_uniform

export @diff, Adadelta, Adagrad, Adam, AutoGrad, Knet, KnetArray, Momentum, Nesterov, Param, RNN, Rmsprop, SGD, Sgd, accuracy, adadelta, adadelta!, adagrad, adagrad!, adam, adam!, batchnorm, bce, bilinear, bmm, bnmoments, bnparams, cat1d, conv4, converge, converge!, cpucopy, deconv4, dropout, elu, gaussian, goldensection, gpu, gpucopy, grad, gradloss, hyperband, invx, ka, knetgc, logistic, logp, logsoftmax, logsumexp, mat, minibatch, momentum, momentum!, nesterov, nesterov!, nll, optimizers, param, param0, params, pool, progress, progress!, relu, rmsprop, rmsprop!, rnninit, rnnparam, rnnparams, selu, setseed, sgd, sgd!, sigm, softmax, train!, training, unpool, update!, value, xavier, xavier_normal, xavier_uniform, zeroone

# This is assumed by some old scripts:
export rnnforw

end # module


