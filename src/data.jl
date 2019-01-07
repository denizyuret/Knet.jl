using Random
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle

"Minibatched data"
mutable struct Data; x; y; batchsize; length; partial; indices; shuffle; xsize; ysize; xtype; ytype; end

"""
    minibatch(x, [y], batchsize; shuffle, partial, xtype, ytype, xsize, ysize)

Return an iterable of minibatches [(xi,yi)...] given data tensors x, y
and batchsize.  The last dimension of x and y should match and give
the number of instances. `y` is optional.  Keyword arguments:

- `shuffle=false`: Shuffle the instances before minibatching.
- `partial=false`: If true include the last partial minibatch < batchsize.
- `xtype=typeof(x)`: Convert xi in minibatches to this type.
- `ytype=typeof(y)`: Convert yi in minibatches to this type.
- `xsize=size(x)`: Convert xi in minibatches to this shape.
- `ysize=size(y)`: Convert yi in minibatches to this shape.
"""
function minibatch(x,y,batchsize; shuffle=false,partial=false,xtype=typeof(x),ytype=typeof(y),xsize=size(x), ysize=size(y))
    nx = size(x)[end]
    if nx != size(y)[end]; throw(DimensionMismatch()); end
    x2 = reshape(x, :, nx)
    y2 = reshape(y, :, nx)
    # indices = shuffle ? randperm(nx) : 1:nx --> do this at the beginning of for loop
    Data(x2,y2,batchsize,nx,partial,1:nx,shuffle,xsize,ysize,xtype,ytype)
end

function minibatch(x,batchsize; shuffle=false,partial=false,xtype=typeof(x),xsize=size(x))
    nx = size(x)[end]
    x2 = reshape(x, :, nx)
    # indices = shuffle ? randperm(nx) : 1:nx --> do this at the beginning of for loop
    Data(x2,nothing,batchsize,nx,partial,1:nx,shuffle,xsize,nothing,xtype,nothing)
end

@propagate_inbounds function iterate(d::Data,i=0)     # returns data in d.indices[i+1:i+batchsize]
    if i == 0 && d.shuffle; d.indices = randperm(d.length); end
    if i >= d.length || (!d.partial && i + d.batchsize > d.length); return nothing; end
    j=min(i+d.batchsize, d.length)
    ids = d.indices[i+1:j]
    xbatch = convert(d.xtype, reshape(d.x[:,ids],d.xsize[1:end-1]...,length(ids)))
    if d.y == nothing
        return (xbatch,j)
    else
        ybatch = convert(d.ytype, reshape(d.y[:,ids],d.ysize[1:end-1]...,length(ids)))
        return ((xbatch,ybatch),j)
    end
end

function eltype(d::Data)
    d.y === nothing ? d.xtype : Tuple{d.xtype, d.ytype}
end

function length(d::Data)
    n = d.length / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

function rand(d::Data)
    i = rand(0:(d.length-d.batchsize))
    return next(d, i)[1]
end

# Use repeat(data,n) for multiple epochs:

struct Repeat; data::Data; n::Int; end

repeat(d::Data, n::Int) = (@assert n >= 0; Repeat(d,n))
length(r::Repeat) = r.n * length(r.data)
eltype(r::Repeat) = eltype(r.data)
eltype(c::Cycle{Data}) = eltype(c.xs)
eltype(c::Cycle{Repeat}) = eltype(c.xs)

@propagate_inbounds function iterate(r::Repeat, s=(1,))
    epoch, state = s[1], tail(s)
    epoch > r.n && return nothing
    next = iterate(r.data, state...)
    next === nothing && return iterate(r, (epoch+1,))
    (next[1], (epoch, next[2]))
end
