import Knet.Ops20: softmax, logsoftmax
using Knet.Ops20: _softmax, _logsoftmax
using Knet.KnetArrays: DevArray
using AutoGrad: AutoGrad, @primitive1
using CUDA.CUDNN: CUDNN, cudnnSoftmaxForward, cudnnSoftmaxBackward #, handle
using CUDA.CUDNN: CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE
#include("cudnn_retry.jl") # @cudnn_retry


function logsoftmax(x::R; dims=:) where {T,R<:DevArray{T}}
    if dims isa Colon || (ndims(x) === 1 && (1 in dims)) # TODO: compare speed to fallback when n>20000
        _x = reshape(x, (1,1,length(x),1))
        _y = _cudnnSoftmaxForward(_x, algo=CUDNN_SOFTMAX_LOG)
        y = reshape(_y, size(x))
    elseif unique(dims) == [1]
        _x = reshape(x, (1,1,size(x,1),:))
        _y = _cudnnSoftmaxForward(_x, algo=CUDNN_SOFTMAX_LOG)
        y = reshape(_y, size(x))
    elseif ndims(x) == 2 && unique(dims) == [2]
        y = permutedims(logsoftmax(permutedims(x); dims=1))
    else
        y = _logsoftmax(x; dims=dims)
    end
    return y
end

function softmax(x::R; dims=:) where {T,R<:DevArray{T}}
    if dims isa Colon || (ndims(x) === 1 && (1 in dims)) # TODO: compare speed to fallback when n>20000
        _x = reshape(x, (1,1,length(x),1))
        _y = _cudnnSoftmaxForward(_x, algo=CUDNN_SOFTMAX_ACCURATE)
        y = reshape(_y, size(x))
    elseif unique(dims) == [1]
        _x = reshape(x, (1,1,size(x,1),:))
        _y = _cudnnSoftmaxForward(_x, algo=CUDNN_SOFTMAX_ACCURATE)
        y = reshape(_y, size(x))
    elseif ndims(x) == 2 && unique(dims) == [2]
        y = permutedims(softmax(permutedims(x); dims=1))
    else
        y = _softmax(x; dims=dims)
    end
    return y
end

function _cudnnSoftmaxForward(x::R; algo) where {T,R<:DevArray{T}}
    mode = CUDNN_SOFTMAX_MODE_INSTANCE
    y = similar(x)
    @cudnn_retry unsafe_cudnnSoftmaxForward(CUDNN.handle(), algo, mode, Ref(T(1)), TD4(x), x, Ref(T(0)), TD4(y), y)
    return y
end

function _cudnnSoftmaxBackward(y::R, dy::R; algo) where {T,R<:DevArray{T}}
    mode = CUDNN_SOFTMAX_MODE_INSTANCE
    dx = similar(y)
    @cudnn_retry unsafe_cudnnSoftmaxBackward(CUDNN.handle(), algo, mode, Ref(T(1)), TD4(y), y, TD4(dy), dy, Ref(T(0)), TD4(dx), dx)
    return dx
end

@primitive1 _cudnnSoftmaxForward(x;o...),dy,y _cudnnSoftmaxBackward(y,dy;o...)
@primitive1 _cudnnSoftmaxBackward(y,dy;o...),ddx,dx csb1(y,dy,dx,ddx;o...) csb2(y,dy,dx,ddx;o...)

function csb1(y,dy,dx,ddx;algo)
    cdim = ndims(y) - 1
    dims = ((1:cdim)...,) # (mode == 0 ? ((1:cdim)...,) : (cdim,))
    if algo==CUDNN_SOFTMAX_FAST || algo==CUDNN_SOFTMAX_ACCURATE
        ddx .* dy - dy .* sum(y .* ddx, dims=dims) - ddx .* sum(y .* dy, dims=dims) 
    elseif algo==CUDNN_SOFTMAX_LOG
        -ddx .* exp.(y) .* sum(dy,dims=dims)
    else
        error("Unknown algo: $algo")
    end
end

function csb2(y,dy,dx,ddx;algo)
    cdim = ndims(y) - 1
    dims = ((1:cdim)...,) # (mode == 0 ? ((1:cdim)...,) : (cdim,))
    if algo==CUDNN_SOFTMAX_FAST || algo==CUDNN_SOFTMAX_ACCURATE
        y .* (ddx .- sum(y .* ddx, dims=dims))
    elseif algo==CUDNN_SOFTMAX_LOG
        ddx .- sum(ddx .* exp.(y), dims=dims)
    else
        error("Unknown algo: $algo")
    end
end

function TD4(x::DevArray)
    d = ndims(x)
    if d == 4 || d == 5
        TD(x)
    else
        n = size(x,d)
        m = div(length(x),n)
        TD(reshape(x,(1,1,m,n)))
    end
end

#=
mutable structdef enum
{
    CUDNN_SOFTMAX_FAST     = 0,         /* straightforward implementation */
    CUDNN_SOFTMAX_ACCURATE = 1,         /* subtract max from every point to avoid overflow */
    CUDNN_SOFTMAX_LOG      = 2          /* this was introduced at cudnnVersion 3000 */
} cudnnSoftmaxAlgorithm_t;

mutable structdef enum
{
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,   /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL = 1     /* compute the softmax over all C for each H, W, N */
} cudnnSoftmaxMode_t;

=#          

