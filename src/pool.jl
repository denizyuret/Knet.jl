type Pool <: Layer; pd; x; y; dx; dy; 
    Pool(d::Int)=new(PoolingDescriptor((d,d)))
end

# TODO: generalize to 3-D
# TODO: cpu implementation

function forw(l::Pool, x::CudaArray; o...)
    l.x = x
    chksize(l, :y, l.x, cudnnGetPoolingNdForwardOutputDim(l.pd, l.x))
    cudnnPoolingForward(l.pd, l.x, l.y)
end

function back(l::Pool, dy::CudaArray; o...)
    l.dy = dy
    @assert size(l.dy) == size(l.y)
    chksize(l, :dx, l.x)
    cudnnPoolingBackward(l.pd, l.y, l.dy, l.x, l.dx)
end
