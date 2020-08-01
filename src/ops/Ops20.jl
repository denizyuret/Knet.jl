module Ops20
import Knet, AutoGrad
using AutoGrad: @primitive, @zerograd, value

include("activation.jl"); export relu, selu, elu, sigm, invx
include("dropout.jl"); export dropout
include("bmm.jl"); export bmm
include("conv.jl"); export conv4, pool, unpool, deconv4, mat
include("loss.jl"); export logp, softmax, logsoftmax, logsumexp, nll, accuracy, zeroone, logistic, bce
#TODO: split rnn knetarray and cpu implementations.
#TODO: split rnn functional and layer implementations.
#TODO include("batchnorm.jl")

end
