using CUDA
import ..Ops20

# TODO: refactor this code, avoid repetition with knetarrays

function Ops20.generic_softmax(x::T,algo::Int,fallback;dims=:) where T<:Union{<:CuArray, AutoGrad.Value{<:CuArray}}
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

function dimvec(x, dims)
     sz = size(x)
     dims = dims == Colon() ? sz : dims
     sort(union(dims)),sz  # handles duplicate dimensions and integer/vector/tuple dims
end

function cudnnSoftmaxForward(x::CuArray{T}; algo=0, mode=0, alpha=1, handle=CUDNN.handle()) where {T}
    beta = 0 # nonzero beta does not make sense when we create y
    y = similar(x)
    CUDNN.cudnnSoftmaxForward(x, y; 
                              algorithm=CUDNN.cudnnSoftmaxAlgorithm_t(algo),
                              mode=CUDNN.cudnnSoftmaxMode_t(mode),
                              alpha=alpha, beta=beta)
    return y
end

function cudnnSoftmaxBackward(y::CuArray{T}, dy::CuArray{T}; algo=0, mode=0, alpha=1, handle=CUDNN.handle()) where {T}
    beta = 0
    dx = similar(dy)
    CUDNN.cudnnSoftmaxBackward(y, dy, dx;
                               algorithm=CUDNN.cudnnSoftmaxAlgorithm_t(algo),
                               mode=CUDNN.cudnnSoftmaxMode_t(mode),
                               alpha=alpha, beta=beta)
    return dx
end

@primitive cudnnSoftmaxForward(x::CuArray;o...),dy,y cudnnSoftmaxBackward(y,dy;o...)
@primitive cudnnSoftmaxBackward(y::CuArray,dy::CuArray;o...),ddx,dx csb1(y,dy,dx,ddx;o...) csb2(y,dy,dx,ddx;o...)

function csb1(y::CuArray,dy::CuArray,dx::CuArray,ddx;algo=0::CuArray,mode=0,o...)
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

function csb2(y::CuArray,dy::CuArray,dx::CuArray,ddx;algo=0,mode=0,o...)
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

function Ops20.nll(y,a::CuArray{<:Integer}; dims=1, average=true)
    @warn "nll(scores, answers::CuArray{<:Integer} is inefficient, nll(scores, answers::Array{<:Integer}) is better." maxlog=1
    nll(y, Array(a); dims=dims, average=average)
end

