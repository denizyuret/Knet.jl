# squeeze any allowed dims argument into a sorted vec with no repetitions
function dimvec(x, dims)
    sz = size(x)
    dims = dims == Colon() ? size(x) : dims
    sort(union(dims))  # handles duplicate dimensions and integer/vector/tuple dims
end

"""
    logp(x; dims=:)

Treat entries in `x` as as unnormalized log probabilities and return
normalized log probabilities.
`dims` is an optional argument, if not specified the normalization is
over all of the dimension of `x`, otherwise the normalization is performed over the
given dimensions.  In particular, if `x` is a matrix, `dims=1`
normalizes columns of `x`, `dims=2` normalizes rows of `x` and
`dims=(1,2)` or `dims=:` normalizes the whole matrix.  
Calls to `logsoftmax` are equivalent to  calls `logp`, and `softmax(x)` is 
equivalent to `exp.(logp(x)`. 
"""
logp(x; dims=:) = _logp(x; dims=dims)

function logp(x::A; dims=:) where A <: Union{<:KnetArray, Param{<:KnetArray}}
    d = dimvec(x, dims)
    if ndims(x) == length(d) # normalizing over all dimensions
        n = length(x)		 
        if n > 20000 
            _logp(x, dims=dims)
        else
            x = cudnnSoftmaxForward(reshape(x,(1,1,n,1)),algo=2)
            reshape(x, sz)
        end 
    elseif d == [1]
        x = cudnnSoftmaxForward(reshape(x, (1,1,sz[1],:)), algo=2)
        reshape(x, sz)
    elseif ndims(x) == 2 && d == [2]
        logp(x', dims=1)' 
    else
        _logp(x, dims=dims)
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
function _logp(x; dims)
    xval = value(x)
    if isa(xval,Number)
        return zero(xval)
    elseif isempty(xval)
        return xval
    else
        x = x .- maximum(x,dims=dims)
        return (x .- log.(sum(exp.(x),dims=dims)))
        # Expanding for profiling:
        # x1 = maximum(x,dims)
        # x2 = x .- x1
        # x3 = exp.(x2)
        # x4 = sum(x3,d...)
        # x5 = log.(x4)
        # x6 = x2 .- x5
        # return x6
    end
end

function _logpback(x,y,dy; dims)
    xval = value(x)
    if isa(xval,Number)
        return zero(xval)
    elseif isempty(xval)
        return xval
    else
        return (dy .- exp.(y).*sum(dy;dims=dims))
        # Expanding for profiling:
        # dx1 = sum(dy,dims)
        # dx2 = exp.(y)
        # dx3 = dx2 .* dx1
        # dx4 = dy - dx3
        # return dx4
    end
end

# dy should be -p and y=logq so this should give us -p+q
@primitive  _logp(x; dims=:),dy,y  _logpback(x,y,dy,dims=dims)

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
softmax(x; dims=:, algo=1) = _softmax(x; dims=dims, algo=algo) # generic fallback

function softmax(x::A; dims=:, algo=1) where A <: Union{<:KnetArray, Param{<:KnetArray}}
    @assert algo ∈ [0, 1]
    d = dimvec(x, dims)
    if ndims(x) == length(d)
        x = cudnnSoftmaxForward(reshape(x,(1,1,:,1)),algo=algo)
        reshape(x, sz)
    elseif d == [1]
        x = cudnnSoftmaxForward(reshape(x, (1,1,sz[1],:)), algo=algo)
        reshape(x, sz)
    elseif ndims(x) == 2 && d == [2]
        softmax(x', dims=1, algo=algo)'
    else
        _softmax(x; dims=dims, algo=algo)
    end
end 

# softmax fallback
function _softmax(x; dims, algo)
    @assert algo ∈ [0, 1]
    if algo == 1
        x = x .- maximum(x, dims=dims)
    end    
    x = exp.(x)
    return x ./ sum(x, dims=dims)
end

function _softback(x,y,dy,dims)
    return y .* dy .- y .* sum(y .* dy, dims=dims)
end

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
          handle, algo, mode, Ref(T(alpha)), TD(x), x, Ref(T(beta)), TD(y), y)
    return y
end

function cudnnSoftmaxBackward(y::KnetArray{T}, dy::KnetArray{T}; algo=0, mode=0, alpha=1, handle=cudnnhandle()) where {T}
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

    logsumexp(x; dims=:)

Compute `log(sum(exp(x);dims))` in a numerically stable manner.

`dims` is an optional argument, if not specified the summation is over
the whole `x`, otherwise the summation is performed over the given
dimensions.  In particular if `x` is a matrix, `dims=1` sums columns
of `x` and `dims=2` sums rows of `x`.
"""
function logsumexp(x; dims=:)
    xmax = maximum(x, dims=dims)
    xmax + log.(sum(exp.(x .- xmax), dims=dims))
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
function nll(y,a::AbstractArray{<:Integer}; dims=1, average=true)
    indices = findindices(y,a,dims=dims)
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
function accuracy(y,a::AbstractArray{<:Integer}; dims=1, average=true)
    indices = findindices(y,a,dims=dims)
    ycpu = Array(y)
    (maxval,maxind) = findmax(ycpu,dims=dims)
    maxind = LinearIndices(ycpu)[maxind]
    correct = (vec(maxind) .== indices)
    average ? mean(correct) : sum(correct)
end

function findindices(y,a::AbstractArray{<:Integer}; dims=1)
    n = length(a)
    indices = Vector{Int}(undef,n)
    if dims == 1                   # instances in first dimension
        y1 = size(y,1)
        y2 = div(length(y),y1)
        if n != y2; throw(DimensionMismatch()); end
        @inbounds for j=1:n
            indices[j] = (j-1)*y1 + a[j]
        end
    elseif dims == 2               # instances in last dimension
        y2 = size(y,ndims(y))
        y1 = div(length(y),y2)
        if n != y1; throw(DimensionMismatch()); end
        @inbounds for j=1:n
            indices[j] = (a[j]-1)*y1 + j
        end
    else
        error("findindices only supports dims = 1 or 2")
    end
    return indices
end



"""
    nll(model, data; dims=1, average=true, o...)

Compute `nll(model(x; o...), y; dims)` for `(x,y)` in `data` and return the per-instance
average (if average=true) or total (if average=false) negative log likelihood.
"""
function nll(model, data; dims=1, average=true, o...)
    sum = cnt = 0
    for (x,y) in data
        sum += nll(model(x; o...), y; dims=dims, average=false)
        cnt += length(y)
    end
    average ? sum / cnt : sum
end


"""
    accuracy(model, data; dims=1, average=true, o...)

Compute `accuracy(model(x; o...), y; dims)` for `(x,y)` in `data` and return the ratio (if
average=true) or the count (if average=false) of correct answers.
"""
function accuracy(model, data; dims=1, average=true, o...)
    sum = cnt = 0
    for (x,y) in data
        sum += accuracy(model(x; o...), y; dims=dims, average=false)
        cnt += length(y)
    end
    average ? sum / cnt : sum
end

zeroone(x...; o...) = 1 - accuracy(x...; o...)

# We need the (model,x,y) interface to implement regularization:
nll(f, x, y; dims=1, average=true, o...)=nll(f(x; o...), y; dims=dims, average=average)
accuracy(f, x, y; dims=1, average=true, o...)=accuracy(f(x; o...), y; dims=dims, average=average)

# We need the (weights,data,predict) interface to support the old interface:
nll(w, data, f::Function; dims=1, average=true, o...)=nll(x->f(w,x;o...), data; dims=dims, average=average)
accuracy(w, data, f::Function; dims=1, average=true, o...)=accuracy(x->f(w,x;o...), data; dims=dims, average=average)
