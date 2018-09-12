# We assume a model is just a callable object (https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1)
# model(x) will give us a prediction, and params(model) will iterate over the parameters.

"""
    param(array; atype)
    param(dims...; init, atype)
    param0(dims...; atype)

The first form returns `Param(atype(array))` where `atype=identity` is the default.

The second form Returns a randomly initialized `Param(atype(init(dims...)))`.
By default, `init` is `xavier` and `atype` is `KnetArray{Float32}` if `gpu() >= 0`,
otherwise `Array{Float32}`. 

The third form `param0` is an alias for `param(dims...; init=zeros)`.
"""
param,param0

param(x::Union{Array,KnetArray}; atype=identity) = Param(atype(x))
param(d...; init=xavier, atype=atype())=Param(atype(init(d...)))
param0(d...; atype=atype())=param(d...; init=zeros, atype=atype)
atype()=(gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

# Keyword argument problem:
# optimizer, loss, model can all take keyword args; how do we specify them through train?
# We can give a constructed optimizer and deepcopy it for each param.
# We don't call model directly, only through loss (because it may need model params for regularization).
# So we pass all unrecognized kwargs to loss and let it sort out.

# What to pass to the callback:
# model, data, loss, optimizer and (o...) are all available to the caller. No need to pass to callback.
# The only things that are not available are J,x,y. I can't think of a use for x,y.
# That leaves J. I considered passing value(J), however that prevents the callback from looking at gradients.
# (e.g. for reporting the gradient norms), so I decided to pass back J as is.


"""
    train!(model, data; loss, optimizer, callback, o...)

Train a model with given data.

* `model`: A callable object. `model(x; o...)` should return a prediction. `params(model)`
   will automatically iterate over model parameters.
* `data`: An iterator. `for (x,y) in data` should iterate over input-output pairs.
* `loss=nll`: A loss function, called with `J = @diff loss(model,x,y; o...)`.
* `optimizer=Adam()`: An optimizer object that will be copied for each parameter and used by
  `[update!]`(@ref).
* `callback`: To facilitate reporting and termination, a callback function is called before
   every update with `callback(J)`. Training continues if the return value is true,
   terminates if it is false.  The default callback runs until training loss convergence.
* Other keyword arguments `(o...)` will be passed to `loss` and possibly by `loss` to `model`.
"""
function train!(model, data; loss=nll, optimizer=Adam(), callback=converge(), o...)
    ps = params(model)
    for param in ps
        param.opt = deepcopy(optimizer)
    end
    while true
        for (x,y) in data
            J = @diff loss(model,x,y; o...)
            if !callback(J)
                return
            end
            for param in ps
                g = grad(J,param)
                update!(value(param),g,param.opt)
            end
        end
    end
end

# import ProgressMeter            # don't want to import update!

function converge(alpha = 0.001)
    avgx = Inf
    avgp = 0.0
    # prog = ProgressMeter.ProgressThresh(0.0, "Training loss: ")
    function callback(x)
        x = value(x)
        if avgx == Inf; avgx = x; end
        p = x - avgx
        avgx = alpha * x + (1-alpha) * avgx
        avgp = alpha * p + (1-alpha) * avgp
        # ProgressMeter.update!(prog,avgx)
        return avgp <= 0.0
    end
    return callback
end


# Issues:
# What if we call train multiple times, and don't want to use the optimizers?
# Do we want parameter initialization as well? init and opt init should happen once.
# Recording losses with different loss functions.
# What info does the callback need?
# Are we doing anything other than pushing kwargs from train to Train?
# What if we want convergence in trnloss or convergence in devloss? Return earlier (best) model?
# How do we easily measure epochs?
# ProgressMeter both in time mode and converge mode.
# Printing loss with ProgressMeter seems difficult.
# Frequency of progress updates and loss calculations?

