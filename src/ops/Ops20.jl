module Ops20

using AutoGrad: AutoGrad, @primitive, @primitive1, value, Param # we should not need Param in ops -- only in rnn.jl
using Base: has_offset_axes, unsafe_convert
using Knet: training, seed!, atype # we should not need atype or xavier in ops -- only in rnn.jl
using LinearAlgebra.BLAS: libblas, @blasfunc
using LinearAlgebra: chkstride1, BlasInt, lmul!
using NNlib: conv, DenseConvDims, maxpool, meanpool, PoolDims, ∇conv_data, ∇conv_filter, ∇maxpool, ∇meanpool
using Random: rand!
using Statistics: mean, var
import Base: show # this will be unnecessary if we don't use structs -- only in rnn.jl

include("activation.jl"); export elu, relu, selu, sigm
include("dropout.jl");    export dropout
include("bmm.jl");        export bmm
include("conv.jl");       export conv4, deconv4, mat, pool, unpool    # rename conv/deconv, deprecate mat?
include("softmax.jl");    export logsoftmax, logsumexp, softmax, logp # deprecate one of logp/logsoftmax?
include("loss.jl");       export accuracy, bce, logistic, nll         # deprecate one of bce/logistic.
include("batchnorm.jl");  export batchnorm, bnmoments, bnparams       # rethink the interface: single struct? no struct?
include("rnn.jl");        export RNN, rnninit, rnnforw, rnnparam, rnnparams # rethink the interface. no struct?

end
