export minimize, minimize!, converge, converge!, train!
import Base: IteratorSize, IteratorEltype, length, size, iterate, eltype
using Base: haslength, tail, @propagate_inbounds, SizeUnknown
using AutoGrad: AutoGrad, @diff, params, grad, value

# progress(minimize(f, ncycle(data,10)))
# A stream (iterator) based implementation: minimize works like map
# taking a stream of args and generating a stream of func values
# except applying gradient based updates to params at each step

#"Example: `minimize(f,ncycle(data,10))`"
minimize(f,d::I,a=Adam(); params=nothing) where {I} = Minimize{I}(d,f,a,params)
minimize!(x...; o...) = for x in minimize(x...; o...); end

struct Minimize{I}; data::I; func; algo; params; end

IteratorSize(::Type{Minimize{I}}) where {I} = IteratorSize(I)
IteratorEltype(::Type{<:Minimize}) = Base.EltypeUnknown()
length(m::Minimize) = length(m.data)
size(m::Minimize,d...) = size(m.data,d...)

@propagate_inbounds function iterate(m::Minimize, s...)
    next = iterate(m.data, s...)
    next === nothing && return nothing
    (args, s) = next
    y = @diff m.func(args...)
    for x in (m.params === nothing ? params(y) : m.params)
        if x.opt === nothing
            x.opt = clone(m.algo)
        end
        update!(x, grad(y,x))
    end
    #Returning the tape is risky: prevents gc and causes >2x slowdown at memory limit
    #return (y,s)
    return (value(y),s)
end

"""
    converge(itr; alpha=0.1)

Return an iterator which acts exactly like `itr`, but quits when values from `itr` stop
decreasing. `itr` should produce numeric values.

It can be used to train a model with the data cycled:

    progress!(converge(minimize(model,cycle(data))))

`alpha` controls the exponential average of values to detect convergence. Here is how
convergence is decided:

    p = x - avgx
    avgx = c.alpha * x + (1-c.alpha) * avgx
    avgp = c.alpha * p + (1-c.alpha) * avgp
    avgp > 0.0 && return nothing

`converge!(...)` is equivalent to `(for x in converge(...) end)`, i.e.  iterates over the
object created by `converge(...)` and returns `nothing`.

"""
converge(iter::I; alpha=0.1) where {I} = Converge{I}(iter, alpha)
converge!(x...; o...) = for x in converge(x...; o...); end

struct Converge{I}; iter::I; alpha::Float64; end

# Converge is large Filter, does not have known size
# length(c::Converge) = length(c.iter)
# size(c::Converge) = size(c.iter)
eltype(c::Converge) = eltype(c.iter)
IteratorEltype(::Type{Converge{I}}) where {I} = IteratorEltype(I)
IteratorSize(::Type{<:Converge}) = SizeUnknown()

@propagate_inbounds function iterate(c::Converge, s=(0.0,Inf))
    avgp,avgx,state = s[1],s[2],tail(tail(s))
    next = iterate(c.iter, state...)
    next === nothing && return nothing
    (item, state) = next
    x = value(item)
    if avgx == Inf; avgx = x; end
    p = x - avgx
    avgx = c.alpha * x + (1-c.alpha) * avgx
    avgp = c.alpha * p + (1-c.alpha) * avgp
    avgp > 0.0 && return nothing
    (item, (avgp, avgx, state))
end

### DEPRECATED:

"""
    train!(model, data; loss, optimizer, callback, o...)

Train a model with given data. This function is deprecated, please use `sgd`, `adam` etc.

* `model`: A callable object. `model(x; o...)` should return a prediction. `params(model)`
   will automatically iterate over model parameters.
* `data`: An iterator. `for (x,y) in data` should iterate over input-output pairs.
* `loss=nll`: A loss function, called with `J = @diff loss(model,x,y; o...)`.
* `optimizer=Adam()`: An optimizer object that will be copied for each parameter and used by
  `[update!]`(@ref).
* `callback`: To facilitate reporting and termination, a callback function is called before
   every update with `callback(J)`. Training continues if the return value is true, terminates
   if it is false. By default training will end after one pass over the data.
* Other keyword arguments `(o...)` will be passed to `loss` and possibly by `loss` to `model`.
"""
function train!(model, data; loss=nll, optimizer=Adam(), callback=epochs(data,1), o...)
    @warn "train! is deprecated, use sgd!, adam! etc. instead." maxlog=1
    ps = params(model)
    for param in ps
        if param.opt === nothing
            param.opt = clone(optimizer)
        end
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

# """
# Pre-defined callback function constructors:

# * converge(): Trains until convergence
# * updates(n): Stops after n updates
# * epochs(data,n): Trains for n epochs, equivalent to updates(n*length(data))
# """
# converge, updates, epochs

function converge(alpha::Number = 0.001)
    avgx = Inf
    avgp = 0.0
    # prog = Progress()
    function callback(x)
        x = value(x)
        if avgx == Inf; avgx = x; end
        p = x - avgx
        avgx = alpha * x + (1-alpha) * avgx
        avgp = alpha * p + (1-alpha) * avgp
        # display_progress!(prog, x)
        return avgp <= 0.0
    end
    return callback
end

function updates(n)
    # p = Progress(n)
    function callback(x)
        # display_progress!(p, value(x))
        n -= 1
        return n > 0
    end
end

epochs(d,n)=updates(n*length(d))


# # Iterator version:
# "Example: `progress!(train(f,repeat(data,10)))`"
# train(pred, data::I; loss=nll, optimizer=Adam(), callback=nothing, params=nothing, kw...) where {I} = Train{I}(data,pred,loss,optimizer,callback,params,kw,Any)

# # Let's not overwrite old train! for backward compatibility
# #train!(x...; o...) = for x in train(x...; o...); end

# struct Train{I}; data::I; pred; loss; optimizer; callback; params; kw; eltype; end

# length(c::Train) = length(c.data)
# size(c::Train) = size(c.data)
# eltype(c::Train) = (c.eltype === Any ? (c.eltype=typeof(@diff c.loss(c.pred,first(c.data)...;c.kw...))) : c.eltype)
# IteratorSize(::Type{Train{I}}) where {I} = IteratorSize(I)
# IteratorEltype(::Type{<:Train}) = Base.HasEltype()

# @propagate_inbounds function iterate(m::Train, s...)
#     next = iterate(m.data, s...)
#     next === nothing && return nothing
#     (args, s) = next
#     y = @diff m.loss(m.pred, args...; m.kw...)
#     m.callback !== nothing && !m.callback(y) && return nothing
#     for x in (m.params === nothing ? params(y) : m.params)
#         if x.opt === nothing
#             x.opt = clone(m.optimizer)
#         end
#         update!(x, grad(y,x))
#     end
#     return (value(y),s)
# end

### DEAD CODE:


    ## This may be slightly faster but risky if active params change
    # if m.params === nothing
    #     m.params = params(y, m.algo)
    # end
    # for x in m.params
    #     update!(x, grad(y,x))
    # end

# function AutoGrad.params(y::AutoGrad.Tape, optimizer=nothing)
#     p = Param[]
#     for node in y.list
#         x = node.Value
#         if isa(x, Param)
#             if x.opt === nothing && optimizer !== nothing
#                 x.opt = clone(optimizer)
#             end
#             push!(p, x)
#         end
#     end
#     return p
# end

# # Simpler and more flexible alternative to train!
# # Does not care where model ends loss begins or where params are
# # data may consist of tuples of any number of args
# # Epochs can be set by data iterator (convergence cannot)
# function minimize!(func, data, optimizer=Adam())
#     for args in data
#         y = @diff func(args...)
#         for node in y.list      # breaks abstraction
#             x = node.Value
#             if isa(x, Param)
#                 g = grad(y,x)
#                 if x.opt === nothing; x.opt = clone(optimizer); end
#                 update!(x.value, g, x.opt)
#             end
#         end
#     end
# end

# "Returns an iterator over Params on Tape."
# struct Params; tape::AutoGrad.Tape; end

# eltype(::Type{Params}) = Param
# IteratorEltype(::Type{Params}) = HasEltype()
# IteratorSize(::Type{Params}) = SizeUnknown()

# @propagate_inbounds function iterate(p::Params, s::Int=1)
#     next = iterate(p.tape.list, s)
#     while next !== nothing
#         (n,s) = next
#         if isa(n.Value,Param)
#             return (n.Value,s)
#         end
#         next = iterate(p.tape.list, s)
#     end
#     nothing
# end

# # Alternative simpler definition:
# params(t::Tape) = (n.Value for n in t.list if n.Value isa Param)

### DEAD CODE


### Issues:
# + What if we call train multiple times, and don't want to use the optimizers?
# x Do we want parameter initialization as well? init and opt init should happen once.
# - Recording losses with different loss functions.
# x What info does the callback need?
# - Are we doing anything other than pushing kwargs from train to Train?
# - What if we want convergence in trnloss or convergence in devloss? Return earlier (best) model?
# + How do we easily measure epochs?
# + ProgressMeter both in time mode and converge mode.
# + Printing loss with ProgressMeter seems difficult.
# + Frequency of progress updates and loss calculations?

# + Keyword argument problem:
# - optimizer, loss, model can all take keyword args; how do we specify them through train?
# + We can give a constructed optimizer and clone it for each param.
# ? We don't call model directly, only through loss (because it may need model params for regularization).
# ? So we pass all unrecognized kwargs to loss and let it sort out.

# x What to pass to the callback:
# x model, data, loss, optimizer and (o...) are all available to the caller. No need to pass to callback.
# x The only things that are not available are J,x,y. I can't think of a use for x,y.
# x That leaves J. I considered passing value(J), however that prevents the callback from looking at gradients.
# + (e.g. for reporting the gradient norms), so I decided to pass back J as is.

# x We assume a model is just a callable object (https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1)
# x model(x) will give us a prediction, and params(model) will iterate over the parameters.

# + 20190105: Do we even need to assume this? train! can simply look at the Tape to find the
# + parameters! In that case optimizers would need to be set elsewhere.

# x use HasLength after data
# x converge may not have length?
# + first efficiency of iterating y.list
# x separate Param in Knet?

# + write train(model,data) iterator style
# + fix update between display_progress and progress
# x progress should handle HasLength
# + use tape iter in train
# + write params tape as iterator
# - check regularization: 
#     do we need opt args?
#     regularizer as parametric fn?
#     regularizer as part of optimizer?
# - write docs
# x use throttle?

# + use cycle for repeat
# + use take for updates: take(cycle(data),n)
# + shuffling during repeats?
# x filter for params(tape) and converge?
# + make params an optional argument
