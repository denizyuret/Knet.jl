# Knet.Ops20: the Ops20 operator set for deep learning

Knet operators circa 2020 are collected in the Ops20 submodule.  Operators are stateless
functions that extend the base language with machine learning specific functions such as
convolutions, losses, activation functions etc. Other operator sets (e.g. NNlib, Keras etc)
can live in their own submodules. Model implementations that use a specific operator module
are hopefully guaranteed not to break in the future as long as they import
e.g. Knet.Ops20. Instead of breaking Ops20, I will create an Ops21.

This submodule has documentation, generic typeless implementations and gradient
definitions. Ops20 functions can/should have array-type specific implementations in
array-type specific submodules.  KnetArray and CuArray implementations are in Knet.Ops20_gpu
(which does not add any functions itself, only provides GPU implementations). Some
implementations may be imported from other packages such as NNlib.

The following principles guide the API design:

* Choose a minimal set of functions that enable efficient implementations of SOTA models.
* Prefer functional programming: weights and state are passed as arguments and returned as values.
* Prefer keyword arguments, leave large structs with weights and state to Layers.
* Stay close to the CUDNN API.

Exported function list:

* activation: elu, relu, selu, sigm
* softmax: logsoftmax, logsumexp, softmax
* loss: accuracy, bce, logistic, nll
* conv: conv4, deconv4, mat, pool, unpool
* batchnorm: batchnorm, bnmoments, bnparams
* rnn: RNN, rnninit, rnnforw, rnnparam, rnnparams
* misc: dropout, bmm


* todo: batchnorm
    bnmoments
    bnparams
    cat1d
    invx
    logp
    RNN (should go to layers, export rnnforw again)
    rnninit
    rnnparam
    rnnparams
