Knet operators circa 2020 should be collected in an Ops20 submodule. This directory has
documentation, generic implementations and gradient definitions. KnetArray and CuArray
implementations should go to knetarrays and cuarrays folders. Some implementations may be
imported from other packages, e.g. NNlib.

    accuracy
    batchnorm
    bce
    bilinear
    bmm
    bnmoments
    bnparams
    cat1d
    conv4
    deconv4
    dropout
    elu
    gaussian
    invx
    logistic
    logp
    logsoftmax
    logsumexp
    mat
    nll
    pool
    relu
    RNN -> should go to layers, export rnnforw again
    rnninit
    rnnparam
    rnnparams
    selu
    sigm
    softmax
    unpool
    xavier
    xavier_uniform
    xavier_normal
    zeroone
