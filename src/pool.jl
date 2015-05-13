type Pool <: Layer; pd; x; y; dx; dy; 
    Pool(d::Int)=new(PoolingDescriptor((d,d)))
end

# TODO: generalize to 3-D
# TODO: cpu implementation

function forw(l::Pool, x::CudaArray; o...)
    initforw(l, x)
    cudnnPoolingForward(l.pd, l.x, l.y)
end

function initforw(l::Pool, x::CudaArray)
    l.x = x
    chksize(l, :y, l.x, cudnnGetPoolingNdForwardOutputDim(l.pd, l.x))
end

function back(l::Pool, dy::CudaArray; dx=true, o...)
    dx || return
    initback(l, dy)
    cudnnPoolingBackward(l.pd, l.y, l.dy, l.x, l.dx)
end

function initback(l::Pool, dy::CudaArray)
    if (size(dy) == size(l.y))
        l.dy = dy
    else
        @assert length(dy) == length(l.y)
        l.dy = reinterpret(eltype(dy), dy, size(l.y))
    end
    chksize(l, :dx, l.x)
end