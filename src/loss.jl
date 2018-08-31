"""

    logp(x;[dims])

Treat entries in `x` as as unnormalized log probabilities and return
normalized log probabilities.

`dims` is an optional argument, if not specified the normalization is
over the whole `x`, otherwise the normalization is performed over the
given dimensions.  In particular, if `x` is a matrix, `dims=1`
normalizes columns of `x` and `dims=2` normalizes rows of `x`.

"""
function logp(x;dims=:)
    if isa(value(x), KnetArray)
        if dims==(1,)
            cudnnSoftmaxForward(x,algo=2)
        elseif dims==(2,)
            cudnnSoftmaxForward(x',algo=2)'
        elseif dims==(:,)
            n = length(x)
            (n > 20000 ? _logp(x) : # see Knet/prof/softmax.jl for timing info
             reshape(cudnnSoftmaxForward(reshape(x,(1,1,n,1)),algo=2),size(x)))
        else
            _logp(x,dims=dims)
        end
    else
        _logp(x,dims=dims) # fall back on old implementation if not KnetArray
    end
end

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

# We keep the old implementation _logp for CPU arrays, slow cases and
# cases of d not handled by cudnn.
function _logp(x;dims=:)
    xval = value(x)
    if isa(xval,Number)
        return zero(xval)
    elseif isempty(xval)
        return xval
    else
        x = x .- maximum(x,dims=dims)
        return (x .- log.(sum(exp.(x),dims=dims)))
        # Expanding for profiling:
        # x1 = maximum(x,d...)
        # x2 = x .- x1
        # x3 = exp.(x2)
        # x4 = sum(x3,d...)
        # x5 = log.(x4)
        # x6 = x2 .- x5
        # return x6
    end
end

function _logpback(x,y,dy;dims)
    xval = value(x)
    if isa(xval,Number)
        return zero(xval)
    elseif isempty(xval)
        return xval
    else
        return (dy - exp.(y).*sum(dy;dims=dims))
        # Expanding for profiling:
        # dx1 = sum(dy,d...)
        # dx2 = exp.(y)
        # dx3 = dx2 .* dx1
        # dx4 = dy - dx3
        # return dx4
    end
end

# dy should be -p and y=logq so this should give us -p+q
@primitive  _logp(x;dims=:),dy,y  _logpback(x,y,dy,dims=dims)

#=
mutable structdef enum
{
    CUDNN_SOFTMAX_FAST     = 0,         /* straightforward implementation */
    CUDNN_SOFTMAX_ACCURATE = 1,         /* subtract max from every point to avoid overflow */
    CUDNN_SOFTMAX_LOG      = 2
} cudnnSoftmaxAlgorithm_t;

mutable structdef enum
{
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,   /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL = 1     /* compute the softmax over all C for each H, W, N */
} cudnnSoftmaxMode_t;

=#          

function cudnnSoftmaxForward(x::KnetArray{T}; algo=0, mode=0, alpha=1, handle=cudnnhandle()) where {T}
    beta = 0 # nonzero beta does not make sense when we create y
    y = similar(x)
    @cuda(cudnn, cudnnSoftmaxForward,
          (Cptr, Cint, Cint, Ptr{T}, Cptr, Ptr{T}, Ptr{T}, Cptr, Ptr{T}),
          handle, algo, mode, Ref(T(alpha)), TD4(x), x, Ref(T(beta)), TD4(y), y)
    return y
end

function cudnnSoftmaxBackward(y::KnetArray{T}, dy::KnetArray{T}; algo=0, mode=0, alpha=1, handle=cudnnhandle()) where {T}
    beta = 0
    dx = similar(dy)
    @cuda(cudnn, cudnnSoftmaxBackward,
          (Cptr, Cint, Cint, Ptr{T}, Cptr, Ptr{T}, Cptr, Ptr{T}, Ptr{T}, Cptr, Ptr{T}),
          handle, algo, mode, Ref(T(alpha)), TD4(y), y, TD4(dy), dy, Ref(T(beta)), TD4(dx), dx)
    return dx
end

@primitive cudnnSoftmaxForward(x;o...),dy,y cudnnSoftmaxBackward(y,dy;o...)
@zerograd cudnnSoftmaxBackward(y,dy;o...)

function TD4(x::KnetArray)
    d = ndims(x)
    if d == 4 || d == 5
        TD(x)
    else
        n = size(x,d)
        m = div(length(x),n)
        TD(reshape(x,(1,1,m,n)))
    end
end


"""

    logsumexp(x;dims=:)

Compute `log(sum(exp(x);dims))` in a numerically stable manner.

`dims` is an optional argument, if not specified the summation is over
the whole `x`, otherwise the summation is performed over the given
dimensions.  In particular if `x` is a matrix, `dims=1` sums columns
of `x` and `dims=2` sums rows of `x`.

"""
function logsumexp(x;dims=:)
    xmax = maximum(x,dims=dims)
    xmax + log.(sum(exp.(x .- xmax),dims=dims))
end

@primitive logsumexp(x;dims=:),dy,y  (dy .* exp.(x .- y))


"""

    nll(scores, answers; dims=1, average=true)

Given an unnormalized `scores` matrix and an `Integer` array of
correct `answers`, return the per-instance negative log
likelihood. `dims=1` means instances are in columns, `dims=2` means
instances are in rows.  Use `average=false` to return the sum instead
of per-instance average.

"""
function nll(y,a::Array{T}; dims=1, average=true) where {T<:Integer}
    indices = findindices(y,a,dims)
    lp = logp(y,dims=dims)[indices]
    average ? -mean(lp) : -sum(lp)
end


"""

    accuracy(scores, answers; dims=1, average=true)

Given an unnormalized `scores` matrix and an `Integer` array of
correct `answers`, return the ratio of instances where the correct
answer has the maximum score. `dims=1` means instances are in columns,
`dims=2` means instances are in rows. Use `average=false` to return
the number of correct answers instead of the ratio.

"""
function accuracy(y,a::Array{T}; dims=1, average=true) where {T<:Integer}
    indices = findindices(y,a,dims)
    ycpu = Array(y)
    (maxval,maxind) = findmax(ycpu,dims=dims)
    maxind = LinearIndices(ycpu)[maxind]
    correct = (vec(maxind) .== indices)
    average ? mean(correct) : sum(correct)
end

function findindices(y,a::Array{T},d=1) where {T<:Integer}
    n = length(a)
    indices = Vector{Int}(undef,n)
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



"""
    nll(f, data; average=true)

Compute `nll(f(x), y)` for `(x,y)` in `data` and return the
per-instance average (if average=true) or total (if average=false)
negative log likelihood.

"""
function nll(f,data::Data; average=true)
    sum = cnt = 0
    for (x,y) in data
        sum += nll(f(x),y; average=false)
        cnt += length(y)
    end
    average ? sum / cnt : sum
end


"""
    accuracy(model, data, predict; average=true)

Compute `accuracy(predict(model,x), y)` for `(x,y)` in `data` and
return the ratio (if average=true) or the count (if average=false) of
correct answers.

"""
function accuracy(f,data::Data; average=true)
    sum = cnt = 0
    for (x,y) in data
        sum += accuracy(f(x),y; average=false)
        cnt += length(y)
    end
    average ? sum / cnt : sum
end

zeroone(x...) = 1 - accuracy(x...)

