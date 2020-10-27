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
"""
