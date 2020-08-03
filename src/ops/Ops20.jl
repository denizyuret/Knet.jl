module Ops20

using AutoGrad: AutoGrad, @primitive, @primitive1, value
using Base: has_offset_axes, unsafe_convert
using Knet: training, seed!
using LinearAlgebra.BLAS: libblas, @blasfunc
using LinearAlgebra: chkstride1, BlasInt, lmul!
using NNlib: conv, DenseConvDims, maxpool, meanpool, PoolDims, ∇conv_data, ∇conv_filter, ∇maxpool, ∇meanpool
using Random: rand!

include("activation.jl"); export elu, relu, selu, sigm
include("dropout.jl");    export dropout
include("bmm.jl");        export bmm
include("conv.jl");       export conv4, deconv4, mat, pool, unpool
include("softmax.jl");    export logsoftmax, logsumexp, softmax
include("loss.jl");       export accuracy, bce, logistic, nll

# TODO:
#TODO: complete the tests
#TODO: split rnn knetarray and cpu implementations.
#TODO: split rnn functional and layer implementations.
#TODO include("batchnorm.jl")

end
