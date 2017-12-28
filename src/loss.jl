"""
    logp(x, dims=1)

Treat entries in `x` as as unnormalized log probabilities and return
normalized log probabilities.

`dims` is an optional argument, if not specified the normalization is
over the first dimension of `x`, otherwise the normalization is performed over the
given dimensions.  In particular, if `x` is a matrix, `dims=1`
normalizes columns of `x`, `dims=2` normalizes rows of `x` and
`dims=(1,2)` normalizes the whole matrix.  

Calls to `logsoftmax` are equivalent to `logp`, and `softmax(x,dims)` is 
equivalent to `exp.(logp(x,dims)`. 
"""
logp(x, dims=1) = _logp(x, dims) # generic fallback

function logp(x::A, dims=1) where A <: Union{KnetArray, Rec{KnetArray}}
    d = sortunion(dims)
    if areconsecutives(d)
        sz = size(x)
        x = reshape(x, (prod(sz[1:d[1]-1]), 1, prod(sz[d]), prod(sz[d[end]+1:end])))
        x = cudnnSoftmaxForward(x, mode=1, algo=2)
        reshape(x, sz)
    else
        _logp(x, dims)
    end
end

areconsecutives(d) = all(d[i+1] == d[i]+1 for i=1:length(d)-1)
areconsecutives(d::Number) = true
sortunion(dims) = sort(union(dims))
sortunion(dims::Number) = dims
 

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

function _logp(x, dims)
    xval = getval(x)
    if isa(xval,Number)
        return zero(xval)
    elseif isempty(xval)
        return xval
    else
        x = x .- maximum(x, dims)
        return (x .- log.(sum(exp.(x), dims)))
        # Expanding for profiling:
        # x1 = maximum(x,d...)
        # x2 = x .- x1
        # x3 = exp_dot(x2)
        # x4 = sum(x3,d...)
        # x5 = log_dot(x4)
        # x6 = x2 .- x5
        # return x6
    end
end

function _logpback(x,y,dy,dims)
    xval = getval(x)
    if isa(xval,Number)
        return zero(xval)
    elseif isempty(xval)
        return xval
    else
        return dy .- exp.(y).*sum(dy, dims)
        # Expanding for profiling:
        # dx1 = sum(dy,d...)
        # dx2 = exp_dot(y)
        # dx3 = dx2 .* dx1
        # dx4 = dy - dx3
        # return dx4
    end
end

# dy should be -p and y=logq so this should give us -p+q
@primitive  _logp(x,dims),dy,y  _logpback(x,y,dy,dims)

"""
    logsoftmax(x, dims=1)

Equivalent to `logp(x, dims)`. See also `sotfmax`. 
"""
const logsoftmax = logp

"""
    softmax(x, dims=1; algo=1)

The softmax function typically used in classification.
Gives the same results as to `exp.(logp(x, dims))`. 

If `algo=1` computation is more accurate, if `algo=0` it is 
faster. 

See also `logsoftmax`.
"""
softmax(x, dims=1; algo=1) = _softmax(x, dims; algo=algo) # generic fallback

function softmax(x::A, dims=1; algo=1) where A <: Union{KnetArray, Rec{KnetArray}}
    @assert algo ∈ [0, 1]
    d = sortunion(dims)
    if areconsecutives(d)
        sz = size(x)
        x = reshape(x, (prod(sz[1:d[1]-1]), 1, prod(sz[d]), prod(sz[d[end]+1:end])))
        x = cudnnSoftmaxForward(x, mode=1, algo=algo)
        reshape(x, sz)
    else
        _softmax(x, dims)
    end
end 

function _softmax(x, dims; algo=1)
    @assert algo ∈ [0, 1]
    if algo == 1
        x = x .- maximum(x, dims)
    end    
    x = exp.(x)
    return x ./ sum(x, dims)
end

function _softback(x,y,dy,dims)
    return y .* dy .- y .* sum(y .* dy, dims)
end

@primitive  _softmax(x,dims; algo=1),dy,y  _softback(x,y,dy,dims)


#=
typedef enum
{
    CUDNN_SOFTMAX_FAST     = 0,         /* straightforward implementation */
    CUDNN_SOFTMAX_ACCURATE = 1,         /* subtract max from every point to avoid overflow */
    CUDNN_SOFTMAX_LOG      = 2
} cudnnSoftmaxAlgorithm_t;

typedef enum
{
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,   /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL = 1     /* compute the softmax over all C for each H, W, N */
} cudnnSoftmaxMode_t;

=#          

function cudnnSoftmaxForward{T}(x::KnetArray{T}; algo=0, mode=0, alpha=1, handle=cudnnhandle())
    beta = 0 # nonzero beta does not make sense when we create y
    y = similar(x)
    @cuda(cudnn, cudnnSoftmaxForward,
          (Cptr, Cint, Cint, Ptr{T}, Cptr, Ptr{T}, Ptr{T}, Cptr, Ptr{T}),
          handle, algo, mode, Ref(T(alpha)), TD(x), x, Ref(T(beta)), TD(y), y)
    return y
end

function cudnnSoftmaxBackward{T}(y::KnetArray{T}, dy::KnetArray{T}; algo=0, mode=0, alpha=1, handle=cudnnhandle())
    beta = 0
    dx = similar(dy)
    @cuda(cudnn, cudnnSoftmaxBackward,
          (Cptr, Cint, Cint, Ptr{T}, Cptr, Ptr{T}, Cptr, Ptr{T}, Ptr{T}, Cptr, Ptr{T}),
          handle, algo, mode, Ref(T(alpha)), TD(y), y, TD(dy), dy, Ref(T(beta)), TD(dx), dx)
    return dx
end

@primitive cudnnSoftmaxForward(x;o...),dy,y cudnnSoftmaxBackward(y,dy;o...)
@zerograd cudnnSoftmaxBackward(y,dy;o...)

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

