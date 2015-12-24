"""
@knet function pool(x; window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX)

window, padding, stride can be specified as Ints or as tuples.
"""
type Pool <: Op; window; padding; stride; mode;
    function Pool(;window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX, o...)
        new(window,padding,stride,mode)
    end
end

ninputs(::Pool)=1
canoverwrite(::Pool)=false
back_reads_x(::Pool)=true
back_reads_y(::Pool)=true

forw(p::Pool, x, y; o...)=
    (cudnnPoolingForward(x, y; window=p.window, padding=p.padding, stride=p.stride, mode=p.mode); gpusync(); y)

back(p::Pool, dy, dx; x=nothing, y=nothing, o...)=
    (dx!=nothing && cudnnPoolingBackward(y, dy, x, dx; window=p.window, padding=p.padding, stride=p.stride, mode=p.mode); gpusync())

function infersize(p::Pool,x,y)
    if x==nothing && y==nothing
        nothing
    elseif x==nothing
        infersize(p, ntuple(i->0, length(y)), y)
    elseif y==nothing
        infersize(p, x, ntuple(i->0, length(x)))
    else
        x = [x...]; y = [y...]
        nd = length(x)-2
        pad = psize(p.padding, nd)
        win = psize(p.window, nd)
        str = psize(p.stride, nd)
        for i=1:nd
            if y[i] > 0 && x[i] > 0
                y[i] == 1 + ceil((x[i] + 2*pad[i] - win[i]) / str[i]) || throw(DimensionMismatch())
            elseif x[i] > 0
                y[i] = 1 + ceil((x[i] + 2*pad[i] - win[i]) / str[i])
            elseif y[i] > 0 && str[i] == 1
                x[i] = y[i] - 1 - 2*pad[i] + win[i]
            end
        end
        for i=nd+1:nd+2
            y[i] == x[i] ? nothing :
            y[i] == 0 ? y[i]=x[i] :
            x[i] == 0 ? x[i]=y[i] :
            throw(DimensionMismatch())
        end
        return (tuple(x...), tuple(y...))
    end
end

psize(w, nd)=(isa(w,Integer) ? ntuple(i->w,nd) : length(w) == nd ? w : throw(DimensionMismatch()))

### DEAD CODE:

# # TODO: generalize to 3-D
# # TODO: cpu implementation

# type Pool <: Op; dims; padding; stride; mode; pd; x; y; ybuf; dx; Pool()=new(); end

# params(::Pool)=Any[]
# ninputs(::Pool)=1
# ysize(l::Pool, x)=cudnnGetPoolingNdForwardOutputDim(l.pd, x)
# overwrites(::Pool)=false
# back_reads_x(::Pool)=true
# back_reads_y(::Pool)=true

# function Pool(dims::Dims;
#               padding=tuple(fill(0,length(dims))...),
#               stride=dims,
#               mode=CUDNN_POOLING_MAX)
#     l=Pool()
#     l.dims=dims
#     l.padding=padding; @assert length(padding)==length(dims)
#     l.stride=stride;   @assert length(stride)==length(dims)
#     l.mode=mode
#     l.pd = PoolingDescriptor(dims, padding=padding, stride=stride, mode=mode)
#     return l
# end

# Pool(d::Int, nd::Int=2; o...)=Pool(tuple(fill(d,nd)...); o...)

# function forw(l::Pool, x; y=nothing, o...)
#     l.x = x
#     y = initforw(l, x, y)
#     cudnnPoolingForward(l.pd, x, y)
#     similar!(l,:y,y); copysync!(l.y,y)
#     return y
# end

# function initforw(l::Pool, x, y)
#     y == nothing && (y = similar!(l, :ybuf, x, ysize(l,x)))
#     typeof(y) == typeof(x) || error("Type mismatch")
#     size(y) == ysize(l,x) || error("Size mismatch")
#     return y
# end

# function back(l::Pool, dy; dx=nothing, x=l.x, y=l.y, returndx=true, o...)
#     returndx || return
#     dx = initback(l, x, y, dx, dy)
#     cudnnPoolingBackward(l.pd, y, dy, x, dx)
# end

# function initback(l::Pool, x, y, dx, dy)
#     issimilar(dy, y) || error("Gradient mismatch in y")
#     dx == nothing && (dx = similar!(l, :dx, x))
#     issimilar(dx, x) || error("Gradient mismatch in x")
#     return dx
# end

# # Make things work with KUdense

# import CUDNN: cudnnGetPoolingNdForwardOutputDim, cudnnPoolingForward, cudnnPoolingBackward

# cudnnGetPoolingNdForwardOutputDim(pd::PoolingDescriptor, x::KUdense)=cudnnGetPoolingNdForwardOutputDim(pd, x.arr)
# cudnnPoolingForward(pd::PoolingDescriptor, x::KUdense, y::KUdense)=(cudnnPoolingForward(pd, x.arr, y.arr);y)
# cudnnPoolingBackward(pd::PoolingDescriptor, y::KUdense, dy::KUdense, x::KUdense, dx::KUdense)=(cudnnPoolingBackward(pd, y.arr, dy.arr, x.arr, dx.arr);dx)


# # Make things work with CPU (for now)

# cudnnGetPoolingNdForwardOutputDim(pd::PoolingDescriptor, x::Array)=cudnnGetPoolingNdForwardOutputDim(pd, CudaArray(x))
# cudnnPoolingForward(pd::PoolingDescriptor, x::Array, y::Array)=(y1=CudaArray(y);cudnnPoolingForward(pd, CudaArray(x), y1); copysync!(y,1,y1,1,length(y)))
# cudnnPoolingBackward(pd::PoolingDescriptor, y::Array, dy::Array, x::Array, dx::Array)=(dx1=CudaArray(dx);cudnnPoolingBackward(pd, CudaArray(y), CudaArray(dy), CudaArray(x), dx1); copysync!(dx,1,dx1,1,length(dx)))

# pool(x,y; window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX, o...)=
#     (Pool(window, padding, stride, mode), x, y)

