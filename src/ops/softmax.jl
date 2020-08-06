export logsoftmax, logsumexp, softmax
using AutoGrad: AutoGrad, @primitive1

"""
    softmax(x; dims=:)
    logsoftmax(x; dims=:)

Treat entries in `x` as as unnormalized log probabilities and return normalized (log)
probabilities, i.e. 

    softmax(x; dims) = exp.(x) ./ sum(exp.(x); dims=dims)
    logsoftmax(x; dims) = x .- log.(sum(exp.(x); dims=dims))

For numerical stability `x = x .- maximum(x,dims=dims)` is performed before exponentiation.

`dims` is an optional argument, if not specified the normalization is over the whole `x`,
otherwise the normalization is performed over the given dimensions.  In particular, if `x` is
a matrix, `dims=1` normalizes columns of `x` and `dims=2` normalizes rows of `x`.

"""
softmax, logsoftmax, logp

"""
    logsumexp(x;dims=:)

Compute `log(sum(exp(x);dims))` in a numerically stable manner.

`dims` is an optional argument, if not specified the summation is over the whole `x`,
otherwise the summation is performed over the given dimensions.  In particular if `x` is a
matrix, `dims=1` sums columns of `x` and `dims=2` sums rows of `x`.

"""
logsumexp


function logsoftmax(x; dims=:)
    isa(value(x),Number) && return zero(x)
    isempty(x) && return x
    x = x .- maximum(x,dims=dims)
    return (x .- log.(sum(exp.(x); dims=dims)))
end

function ∇logsoftmax(x,y,dy; dims)
    isa(value(x),Number) && return zero(x)
    isempty(x) && return x
    return (dy - exp.(y).*sum(dy; dims=dims))
end

function softmax(x; dims=:)
    isa(value(x),Number) && return one(x)
    isempty(x) && return x
    x = x .- maximum(x; dims=dims)
    x = exp.(x)
    return (x ./ sum(x; dims=dims))
end

function ∇softmax(x,y,dy; dims=:)
    isa(value(x),Number) && return zero(x)
    isempty(x) && return x
    return (y .* dy .- y .* sum(y .* dy; dims=dims))
end

function logsumexp(x; dims=:)
    isa(value(x),Number) && return x
    isempty(x) && throw(ArgumentError("reducing over an empty collection is not allowed"))
    xmax = maximum(x,dims=dims)
    xmax + log.(sum(exp.(x .- xmax),dims=dims))
end

function ∇logsumexp(x,y,dy; dims)
    isa(value(x),Number) && return one(x)
    isempty(x) && throw(ArgumentError("reducing over an empty collection is not allowed"))
    return (dy .* exp.(x .- y))
end    

@primitive1  logsoftmax(x;dims=:),dy,y  ∇logsoftmax(x,y,dy,dims=dims)
@primitive1  softmax(x;dims=:),dy,y     ∇softmax(x,y,dy,dims=dims)
@primitive1  logsumexp(x;dims=:),dy,y   ∇logsumexp(x,y,dy;dims=dims)
const logp = logsoftmax


# Math for the softmax loss: x is unnormalized input, p is target probabilities, q is
# estimated probabilities. Read left column down, right column (loss gradients) back up.
#
# x			dx = -p + qz/z = -p + exp(logq)
# xmax  = max(x,1)	-sum(db)=0
# logqz = x .- xmax	-p + qz/z
# qz    = exp(logqz)	rep(1/z)
# z     = sum(qz,1)	1/z
# logz  = log(z)	sum(p)=1
# logq  = logqz.-logz	-p
# plogq = p .* logq	-1
# loss  = -sum(plogq)	1
