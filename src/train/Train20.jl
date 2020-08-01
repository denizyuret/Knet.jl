module Train20
using ..Knet: atype
using AutoGrad
include("distributions.jl")
include("hyperopt.jl")
include("progress.jl")
include("train.jl")
include("update.jl")

# TODO: reduce this list
export
    adadelta!,
    Adadelta,
    adadelta,
    adagrad!,
    Adagrad,
    adagrad,
    adam!,
    Adam,
    adam,
    bilinear,
    converge!,
    converge,
    gaussian,
    goldensection,
    hyperband,
    minimize!,
    minimize,
    momentum!,
    Momentum,
    momentum,
    nesterov!,
    Nesterov,
    nesterov,
    optimizers,
    param,
    param0,
    progress!,
    progress,
    rmsprop!,
    Rmsprop,
    rmsprop,
    sgd!,
    SGD,
    Sgd,
    sgd,
    train!,
    update!,
    xavier,
    xavier_uniform,
    xavier_normal
    
end
