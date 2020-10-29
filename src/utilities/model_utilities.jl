"""
using Printf
using Knet

struct Layer
    w
    b
    f
end

mutable struct Model
    layers::Array{Layer,1}
    loss_func
    opt_func
end

Model(l::Array{Layer,1}; loss_func = nll, optimization_func = adam) = Model(l,loss_func,optimization_func)

In my point of view, Knet is in need of high-level model object and related functions such as model(...), add(model, layer),
summary(model), but I could not be sure whether this is a design choice so I did not work on an implementation. If this is not
a design choice, I am willing to offer my help in implementation of tensorflow.model like object
"""
