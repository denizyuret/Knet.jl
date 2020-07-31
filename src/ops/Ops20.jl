module Ops20
using AutoGrad

include("activation.jl"); export relu, selu, elu, sigm
include("dropout.jl"); export dropout
include("bmm.jl"); export bmm
include("conv.jl"); export conv4, pool, unpool, deconv4
include("loss.jl"); export logp, softmax, logsoftmax, logsumexp, nll, accuracy, zeroone, logistic, bce
# include("rnn.jl")
mutable struct RNN end
# include("batchnorm.jl")
function batchnorm end

end
