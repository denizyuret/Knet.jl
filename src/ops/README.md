Knet operators circa 2020 should be collected in an Ops20 submodule. This directory has
documentation, generic typeless implementations and gradient definitions. KnetArray and
CuArray implementations should go to knetarrays and cuarrays folders. Some implementations may
be imported from other packages, e.g. NNlib.

Functions in Ops and Base can/should have array-type specific implementations. All other
levels (layers, models) should be generic and should work with any cpu/gpu array type
without changing the code.

    batchnorm
    bce
    bilinear
*   bmm
    bnmoments
    bnparams
    cat1d
    conv4
    deconv4
    dropout
*   elu
    invx
    logistic
*   logp
*   logsoftmax
    logsumexp
    mat
    nll
    pool
*   relu
    RNN -> should go to layers, export rnnforw again
    rnninit
    rnnparam
    rnnparams
*   selu
*   sigm
    softmax
    unpool
