type Pool <: Layer; dims; padding; stride; mode; pd; x; y; dx; dy; Pool()=new(); end

# TODO: generalize to 3-D
# TODO: cpu implementation

if GPU

function Pool(dims::(Int...);
              padding=tuple(fill(0,length(dims))...),
              stride=dims,
              mode=CUDNN_POOLING_MAX)
    l=Pool()
    l.dims=dims
    l.padding=padding; @assert length(padding)==length(dims)
    l.stride=stride;   @assert length(stride)==length(dims)
    l.mode=mode
    Atype==CudaArray && (l.pd = PoolingDescriptor(dims, padding=padding, stride=stride, mode=mode))
    return l
end

Pool(d::Int, nd::Int=2; o...)=Pool(tuple(fill(d,nd)...); o...)

copy(l::Pool)=Pool(l.dims; padding=l.padding, stride=l.stride, mode=l.mode)

function forw(l::Pool, x; o...)
    # error("CPU pool not implemented")
    a = KUnet.Atype
    KUnet.atype(CudaArray)
    y = forw(copy(l), CudaArray(x); o...)
    KUnet.atype(a)
    l.x = x
    l.y = to_host(y)
end

function back(l::Pool, dy; o...)
    # error("CPU pool not implemented")
    a = KUnet.Atype
    KUnet.atype(CudaArray)
    ll = copy(l); ll.y = CudaArray(l.y); ll.x = CudaArray(l.x)
    dx = back(ll, CudaArray(dy); o...)
    KUnet.atype(a)
    l.dy = dy
    l.dx = to_host(dx)
end

function forw(l::Pool, x::CudaArray; o...)
    initforw(l, x)
    cudnnPoolingForward(l.pd, l.x, l.y)
end

function initforw(l::Pool, x::CudaArray)
    l.x = x
    similar!(l, :y, l.x, cudnnGetPoolingNdForwardOutputDim(l.pd, l.x))
end

function back(l::Pool, dy::CudaArray; returndx=true, o...)
    returndx || return
    initback(l, dy)
    cudnnPoolingBackward(l.pd, l.y, l.dy, l.x, l.dx)
end

function initback(l::Pool, dy::CudaArray)
    if (size(dy) == size(l.y))
        l.dy = dy
    else
        @assert length(dy) == length(l.y)
        l.dy = reshape(dy, size(l.y))
    end
    similar!(l, :dx, l.x)
end

else
warn("No cpu pool")
Pool(x)=Pool()
copy(l::Pool;o...)=Pool()
forw(l::Pool,x;o...)=(l.x=l.y=x)
back(l::Pool,dy;o...)=(l.dx=l.dy=dy)
end # if GPU
