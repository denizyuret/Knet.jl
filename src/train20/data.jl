export minibatch, Data
import Base: iterate, eltype, length, rand, repeat, summary, show
using Base.Iterators: Cycle
using Base: @propagate_inbounds, tail
using Random: randperm
using Knet: atype

"Iterator of minibatches (x,y pairs) returned by `minibatch`."
mutable struct Data{T}; x; y; batchsize; length; partial; imax; indices; shuffle; xsize; ysize; xtype; ytype; end

"""
    minibatch(x, [y], batchsize; shuffle, partial, xtype, ytype, xsize, ysize)

Return an iterator of minibatches [(xi,yi)...] given data tensors x, y and batchsize.  

The last dimension of x and y give the number of instances and should be equal. `y` is
optional, if omitted a sequence of `xi` will be generated rather than `(xi,yi)` tuples.  Use
`repeat(d,n)` for multiple epochs, `Iterators.take(d,n)` for a partial epoch, and
`Iterators.cycle(d)` to cycle through the data forever (this can be used with `converge`).
If you need the iterator to continue from its last position when stopped early (e.g. by a
break in a for loop), use `Iterators.Stateful(d)` (by default the iterator would restart
from the beginning).

Keyword arguments:

- `shuffle=false`: Shuffle the instances every epoch.
- `partial=false`: If true include the last partial minibatch < batchsize.
- `xtype=typeof(x)`: Convert xi in minibatches to this type.
- `ytype=typeof(y)`: Convert yi in minibatches to this type.
- `xsize=size(x)`: Convert xi in minibatches to this shape (with last dimension adjusted for batchsize).
- `ysize=size(y)`: Convert yi in minibatches to this shape (with last dimension adjusted for batchsize).
"""
minibatch

function minibatch(x,y,batchsize; shuffle=false,partial=false,xsize=size(x),ysize=size(y),
                   # default xtype, ytype should be robust to ndims change:
                   xtype = (eltype(x) <: AbstractFloat ? atype() : (typeof(x).name.wrapper){eltype(x)}),
                   ytype = (eltype(y) <: AbstractFloat ? atype() : (typeof(y).name.wrapper){eltype(y)}))
    nx = size(x)[end]
    if nx != size(y)[end]; throw(DimensionMismatch()); end
    x2 = reshape(x, :, nx)
    y2 = reshape(y, :, nx)
    imax = partial ? nx : nx - batchsize + 1
    Data{Tuple{xtype,ytype}}(x2,y2,batchsize,nx,partial,imax,1:nx,shuffle,xsize,ysize,xtype,ytype)
end

function minibatch(x,batchsize; shuffle=false,partial=false,xsize=size(x),
                   xtype = (eltype(x) <: AbstractFloat ? atype() : (typeof(x).name.wrapper){eltype(x)}))
    nx = size(x)[end]
    x2 = reshape(x, :, nx)
    imax = partial ? nx : nx - batchsize + 1
    Data{xtype}(x2,nothing,batchsize,nx,partial,imax,1:nx,shuffle,xsize,nothing,xtype,nothing)
end

@propagate_inbounds function iterate(d::Data, i=0)     # returns data in d.indices[i+1:i+batchsize]
    if i >= d.imax
        return nothing
    end
    if d.shuffle && i == 0
        d.indices = randperm(d.length)
    end
    nexti = min(i + d.batchsize, d.length)
    ids = d.indices[i+1:nexti]
    xbatch = try convert(d.xtype, reshape(d.x[:,ids],d.xsize[1:end-1]...,length(ids)))
    catch; throw(DimensionMismatch("X tensor not compatible with size=$(d.xsize) and type=$(d.xtype)")); end
    if d.y == nothing
        return (xbatch,nexti)
    else
        ybatch = try convert(d.ytype, reshape(d.y[:,ids],d.ysize[1:end-1]...,length(ids)))
        catch; throw(DimensionMismatch("Y tensor not compatible with size=$(d.ysize) and type=$(d.ytype)")); end
        return ((xbatch,ybatch),nexti)
    end
end

eltype(::Type{Data{T}}) where T = T

function length(d::Data)
    n = d.length / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

function rand(d::Data)
    i = rand(0:(d.length-d.batchsize))
    return iterate(d, i)[1]
end

# IterTools.ncycle(data,n) for multiple epochs
# Base.Iterators.cycle(data) to go forever
# Base.Iterators.take(data,n) for partial epochs
# IterTools.takenth(itr,n) to report every n iterations

struct Repeat; data::Data; n::Int; end

function repeat(d::Data, n::Int)
    @warn "repeat(d::Data,n) is deprecated, use IterTools.ncycle instead." maxlog=1
    @assert n >= 0
    Repeat(d,n)
end

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

# Give length info in summary:
summary(d::Data) = "$(length(d))-element $(typeof(d))"
show(io::IO, d::Data) = print(IOContext(io,:compact=>true), summary(d))
show(io::IO, ::MIME"text/plain", d::Data) = show(io, d)

