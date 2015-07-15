# TODO: generalize to 3-D
# TODO: cpu implementation

type Pool <: Layer; dims; padding; stride; mode; pd; x; y; dx; dy; Pool()=new(); end

if !GPU

warn("No cpu pool")
# Let these give error?
# Pool(x)=Pool()
# copy(l::Pool;o...)=Pool()
# forw(l::Pool,x;o...)=(l.x=l.y=x)
# back(l::Pool,dy;o...)=(l.dx=l.dy=dy)

else

function Pool(dims::(Int...);
              padding=tuple(fill(0,length(dims))...),
              stride=dims,
              mode=CUDNN_POOLING_MAX)
    l=Pool()
    l.dims=dims
    l.padding=padding; @assert length(padding)==length(dims)
    l.stride=stride;   @assert length(stride)==length(dims)
    l.mode=mode
    l.pd = PoolingDescriptor(dims, padding=padding, stride=stride, mode=mode)
    return l
end

Pool(d::Int, nd::Int=2; o...)=Pool(tuple(fill(d,nd)...); o...)

# copy(l::Pool)=Pool(l.dims; padding=l.padding, stride=l.stride, mode=l.mode)

function forw(l::Pool, x::AbstractCudaArray; o...)
    initforw(l, x)
    cudnnPoolingForward(l.pd, l.x, l.y)
end

function initforw(l::Pool, x::AbstractCudaArray)
    l.x = x
    similar!(l, :y, l.x, cudnnGetPoolingNdForwardOutputDim(l.pd, l.x))
end

function back(l::Pool, dy::AbstractCudaArray; returndx=true, o...)
    returndx || return
    initback(l, dy)
    cudnnPoolingBackward(l.pd, l.y, l.dy, l.x, l.dx)
end

function initback(l::Pool, dy::AbstractCudaArray)
    l.dy = ((size(dy) == size(l.y)) ? dy : reshape(dy, size(l.y)))
    similar!(l, :dx, l.x)
end

# function forw(l::Pool, x; o...)
#     # error("CPU pool not implemented")
#     a = KUnet.Atype
#     KUnet.atype(CudaDynArray)
#     y = forw(copy(l), CudaDynArray(x); o...)
#     KUnet.atype(a)
#     l.x = x
#     l.y = to_host(y)
# end

# function back(l::Pool, dy; o...)
#     # error("CPU pool not implemented")
#     a = KUnet.Atype
#     KUnet.atype(CudaDynArray)
#     ll = copy(l); ll.y = CudaDynArray(l.y); ll.x = CudaDynArray(l.x)
#     dx = back(ll, CudaDynArray(dy); o...)
#     KUnet.atype(a)
#     l.dy = dy
#     l.dx = to_host(dx)
# end

end # if GPU
