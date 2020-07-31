using CUDA
import ..Ops20: generic_softmax

# TODO: refactor this code, avoid repetition with cuarrays

function generic_softmax(x::T,algo::Int,fallback;dims=:) where T<:Union{<:KnetArray, AutoGrad.Value{<:KnetArray}}
    d,sz = dimvec(x,dims)
    if algo == 2 && (CUDNN.handle(); CUDNN.version() < v"3") # algo=2 (logsoftmax) was introduced in cudnn 3000
        fallback(x; dims=dims, algo=algo)
    elseif d==[1]
        x = cudnnSoftmaxForward(reshape(x, (1,1,sz[1],:)), algo=algo)
        reshape(x, sz)
    elseif d==[2] && ndims(x)==2
        permutedims(generic_softmax(permutedims(x),algo,fallback;dims=1))
    elseif length(d)==ndims(x);
        n = length(x)
        (n > 20000 ? fallback(x;dims=dims,algo=algo) : # see Knet/prof/softmax.jl for timing info
        reshape(cudnnSoftmaxForward(reshape(x,(1,1,n,1)),algo=algo),size(x)))
    else
        fallback(x;dims=dims,algo=algo)
    end
end


function cudnnSoftmaxForward(x::KnetArray{T}; algo=0, mode=0, alpha=1, handle=CUDNN.handle()) where {T}
    beta = 0 # nonzero beta does not make sense when we create y
    y = similar(x)
    # @cudnn(cudnnSoftmaxForward,
    #       (Cptr, Cint, Cint, Ptr{T}, Cptr, Ptr{T}, Ptr{T}, Cptr, Ptr{T}),
    #       handle, algo, mode, Ref(T(alpha)), TD4(x), x, Ref(T(beta)), TD4(y), y)
    algo = CUDNN.cudnnSoftmaxAlgorithm_t(algo)
    mode = CUDNN.cudnnSoftmaxMode_t(mode)
    CUDNN.cudnnSoftmaxForward(handle, algo, mode, Ref(T(alpha)), TD4(x), x, Ref(T(beta)), TD4(y), y)
    return y
end

function cudnnSoftmaxBackward(y::KnetArray{T}, dy::KnetArray{T}; algo=0, mode=0, alpha=1, handle=CUDNN.handle()) where {T}
    beta = 0
    dx = similar(dy)
    # @cudnn(cudnnSoftmaxBackward,
    #       (Cptr, Cint, Cint, Ptr{T}, Cptr, Ptr{T}, Cptr, Ptr{T}, Ptr{T}, Cptr, Ptr{T}),
    #       handle, algo, mode, Ref(T(alpha)), TD4(y), y, TD4(dy), dy, Ref(T(beta)), TD4(dx), dx)
    algo = CUDNN.cudnnSoftmaxAlgorithm_t(algo)
    mode = CUDNN.cudnnSoftmaxMode_t(mode)
    CUDNN.cudnnSoftmaxBackward(handle, algo, mode, Ref(T(alpha)), TD4(y), y, TD4(dy), dy, Ref(T(beta)), TD4(dx), dx)
    return dx
end

@primitive cudnnSoftmaxForward(x::KnetArray;o...),dy,y cudnnSoftmaxBackward(y,dy;o...)
@primitive cudnnSoftmaxBackward(y::KnetArray,dy::KnetArray;o...),ddx,dx csb1(y,dy,dx,ddx;o...) csb2(y,dy,dx,ddx;o...)

function csb1(y::KnetArray,dy::KnetArray,dx::KnetArray,ddx;algo=0::KnetArray,mode=0,o...)
    cdim = ndims(y) - 1
    dims = (mode == 0 ? ((1:cdim)...,) : (cdim,))
    if algo==0 || algo==1
        ddx .* dy - dy .* sum(y .* ddx, dims=dims) - ddx .* sum(y .* dy, dims=dims) 
    elseif algo==2
        -ddx .* exp.(y) .* sum(dy,dims=dims)
    else
        error("Unknown algo: $algo")
    end
end

function csb2(y::KnetArray,dy::KnetArray,dx::KnetArray,ddx;algo=0,mode=0,o...)
    cdim = ndims(y) - 1
    dims = (mode == 0 ? ((1:cdim)...,) : (cdim,))
    if algo==0 || algo==1
        y .* (ddx .- sum(y .* ddx, dims=dims))
    elseif algo==2
        ddx .- sum(ddx .* exp.(y), dims=dims)
    else
        error("Unknown algo: $algo")
    end
end

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


function nll(y,a::KnetArray{<:Integer}; dims=1, average=true)
    @warn "nll(scores, answers::KnetArray{<:Integer} is inefficient, nll(scores, answers::Array{<:Integer}) is better." maxlog=1
    nll(y, Array(a); dims=dims, average=average)
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

