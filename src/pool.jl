# TODO: generalize to 3-D
# TODO: cpu implementation

type Pool <: Layer; dims; padding; stride; mode; pd; x; y; dx; dy; Pool()=new(); end


function Pool(dims::Dims;
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

function forw(l::Pool, x; o...)
    initforw(l, x)
    cudnnPoolingForward(l.pd, l.x, l.y)
    return l.y
end

function initforw(l::Pool, x)
    l.x = x
    similar!(l, :y, x, cudnnGetPoolingNdForwardOutputDim(l.pd, x))
end

function back(l::Pool, dy; returndx=true, o...)
    returndx || return
    initback(l, dy)
    cudnnPoolingBackward(l.pd, l.y, l.dy, l.x, l.dx)
    return l.dx
end

function initback(l::Pool, dy)
    @assert issimilar(dy, l.y)
    # l.dy = ((size(dy) == size(l.y)) ? dy : reshape(dy, size(l.y)))
    l.dy = dy
    similar!(l, :dx, l.x)
end


# Make things work with KUdense

CUDNN.cudnnGetPoolingNdForwardOutputDim(pd::PoolingDescriptor, x::KUdense)=cudnnGetPoolingNdForwardOutputDim(pd, x.arr)
CUDNN.cudnnPoolingForward(pd::PoolingDescriptor, x::KUdense, y::KUdense)=(cudnnPoolingForward(pd, x.arr, y.arr);y)
CUDNN.cudnnPoolingBackward(pd::PoolingDescriptor, y::KUdense, dy::KUdense, x::KUdense, dx::KUdense)=(cudnnPoolingBackward(pd, y.arr, dy.arr, x.arr, dx.arr);dx)


# Make things work with CPU (for now)

CUDNN.cudnnGetPoolingNdForwardOutputDim(pd::PoolingDescriptor, x::Array)=cudnnGetPoolingNdForwardOutputDim(pd, CudaArray(x))
CUDNN.cudnnPoolingForward(pd::PoolingDescriptor, x::Array, y::Array)=(y1=CudaArray(y);cudnnPoolingForward(pd, CudaArray(x), y1); copy!(y,1,y1,1,length(y)))
CUDNN.cudnnPoolingBackward(pd::PoolingDescriptor, y::Array, dy::Array, x::Array, dx::Array)=(dx1=CudaArray(dx);cudnnPoolingBackward(pd, CudaArray(y), CudaArray(dy), CudaArray(x), dx1); copy!(dx,1,dx1,1,length(dx)))


### DEAD CODE

# else

# warn("No cpu pool")

# end # if GPU

# Let these give error?
# Pool(x)=Pool()
# copy(l::Pool;o...)=Pool()
# forw(l::Pool,x;o...)=(l.x=l.y=x)
# back(l::Pool,dy;o...)=(l.dx=l.dy=dy)

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

