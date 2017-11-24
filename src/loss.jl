"""

    logp(x,[dims])

Treat entries in `x` as as unnormalized log probabilities and return
normalized log probabilities.

`dims` is an optional argument, if not specified the normalization is
over the whole `x`, otherwise the normalization is performed over the
given dimensions.  In particular, if `x` is a matrix, `dims=1`
normalizes columns of `x` and `dims=2` normalizes rows of `x`.

"""

# Math for the cross-entropy loss: x is unnormalized input, p is
# target probabilities, q is estimated probabilities. Read left column
# down, right column (loss gradients) back up.

# x			dx = -p + qz/z = -p + exp(logq)
# xmax  = max(x,1)	-sum(db)=0
# logqz = x .- xmax	-p + qz/z
# qz    = exp(logqz)	rep(1/z)
# z     = sum(qz,1)	1/z
# logz  = log(z)	sum(p)=1
# logq  = logqz.-logz	-p
# plogq = p .* logq	-1
# loss  = -sum(plogq)	1

function logp(x,d...)
    if isa(x,Number)
        return zero(x)
    elseif isempty(x)
        return x
    else
        # x = x .- maximum(x,d...)
        # return (x .- log(sum(exp(x),d...)))
        # Expanding for profiling:
        x1 = maximum(x,d...)
        x2 = x .- x1
        x3 = exp_dot(x2)
        x4 = sum(x3,d...)
        x5 = log_dot(x4)
        x6 = x2 .- x5
        return x6
    end
end

function logpback(x,y,dy,d...)
    x = AutoGrad.getval(x)
    if isa(x,Number)
        return zero(x)
    elseif isempty(x)
        return x
    else
        # return (dy - exp(y).*sum(dy,d...))
        # Expanding for profiling:
        dx1 = sum(dy,d...)
        dx2 = exp_dot(y)
        dx3 = dx2 .* dx1
        dx4 = dy - dx3
        return dx4
    end
end

# dy should be -p and y=logq so this should give us -p+q
@primitive  logp(x,d...),dy,y  logpback(x,y,dy,d...)


"""

    logsumexp(x,[dims])

Compute `log(sum(exp(x),dims))` in a numerically stable manner.

`dims` is an optional argument, if not specified the summation is over
the whole `x`, otherwise the summation is performed over the given
dimensions.  In particular if `x` is a matrix, `dims=1` sums columns
of `x` and `dims=2` sums rows of `x`.

"""
function logsumexp(x,d...)
    xmax = maximum(x,d...)
    xmax + log_dot(sum(exp_dot(x .- xmax),d...))
end

@primitive logsumexp(x,d...),dy,y  (dy .* exp_dot(x .- y))


"""

    nll(scores, answers, d=1; average=true)

Given an unnormalized `scores` matrix and an `Integer` array of
correct `answers`, return the per-instance negative log
likelihood. `d=1` means instances are in columns, `d=2` means
instances are in rows.  Use `average=false` to return the sum instead
of per-instance average.

"""
function nll{T<:Integer}(y,a::Array{T},d=1; average=true)
    indices = findindices(y,a,d)
    lp = logp(y,d)[indices]
    average ? -mean(lp) : -sum(lp)
end


"""

    accuracy(scores, answers, d=1; average=true)

Given an unnormalized `scores` matrix and an `Integer` array of
correct `answers`, return the ratio of instances where the correct
answer has the maximum score. `d=1` means instances are in columns,
`d=2` means instances are in rows. Use `average=false` to return
the number of correct answers instead of the ratio.

"""
function accuracy{T<:Integer}(y,a::Array{T},d=1; average=true)
    indices = findindices(y,a,d)
    (maxval,maxind) = findmax(Array(y),d)
    correct = (vec(maxind) .== indices)
    average ? mean(correct) : sum(correct)
end

function findindices{T<:Integer}(y,a::Array{T},d=1)
    n = length(a)
    indices = Vector{Int}(n)
    if d == 1                   # instances in first dimension
        y1 = size(y,1)
        y2 = div(length(y),y1)
        if n != y2; throw(DimensionMismatch()); end
        @inbounds for j=1:n
            indices[j] = (j-1)*y1 + a[j]
        end
    elseif d == 2               # instances in last dimension
        y2 = size(y,ndims(y))
        y1 = div(length(y),y2)
        if n != y1; throw(DimensionMismatch()); end
        @inbounds for j=1:n
            indices[j] = (a[j]-1)*y1 + j
        end
    else
        error("findindices only supports d = 1 or 2")
    end
    return indices
end



# # The xentloss interface is no good because of double normalization.

# """
# xentloss(x, p [,dims])

# Compute cross-entropy loss for unnormalized log probability estimates
# x and normalized probabilities p normalizing over the given
# dimensions.  By default normalization is over the whole array, dims=1
# normalizes over the columns, dims=2 normalizes over the rows of a 2-D
# array.

# """
# function xentloss(x,p,d...)
#     x = x .- maximum(x,d...)
#     z = log(sum(exp(x),d...))
#     return sum(p .* (x .- z)) / length(z)
# end

# function xentback(x,p,d...)
#     x = x .- maximum(x,d...)
#     x = exp(x)
#     z = sum(x,d...)
#     return x./z - p
# end

# @primitive xentloss(x,p,d...),dy,y  (dy.*xentback(x,p,d...))

