using Random

"Minibatched data"
mutable struct MB; x; y; batchsize; length; partial; indices; xsize; ysize; xtype; ytype; end

"""

    minibatch(x, y, batchsize; shuffle, partial, xtype, ytype)

Return an iterable of minibatches [(xi,yi)...] given data tensors x, y
and batchsize.  The last dimension of x and y should match and give
the number of instances. Keyword arguments:

- `shuffle=false`: Shuffle the instances before minibatching.
- `partial=false`: If true include the last partial minibatch < batchsize.
- `xtype=typeof(x)`: Convert xi in minibatches to this type.
- `ytype=typeof(y)`: Convert yi in minibatches to this type.

"""
function minibatch(x,y,batchsize; shuffle=false,partial=false,xtype=typeof(x),ytype=typeof(y))
    xsize = collect(size(x))
    ysize = collect(size(y))
    nx = xsize[end]
    ny = ysize[end]
    if nx != ny; throw(DimensionMismatch()); end
    x2 = reshape(x, div(length(x),nx), nx)
    y2 = reshape(y, div(length(y),ny), ny)
    indices = shuffle ? randperm(nx) : 1:nx
    MB(x2,y2,batchsize,nx,partial,indices,xsize,ysize,xtype,ytype)
end

"""

    minibatch(x, batchsize; shuffle, partial, xtype, ytype)

Return an iterable of minibatches [x1,x2,...] given data tensor x and
batchsize.  The last dimension of x gives the number of instances.
Keyword arguments:

- `shuffle=false`: Shuffle the instances before minibatching.
- `partial=false`: If true include the last partial minibatch < batchsize.
- `xtype=typeof(x)`: Convert xi in minibatches to this type.

"""
function minibatch(x,batchsize; shuffle=false,partial=false,xtype=typeof(x))
    xsize = collect(size(x))
    nx = xsize[end]
    x2 = reshape(x, div(length(x),nx), nx)
    indices = shuffle ? randperm(nx) : 1:nx
    MB(x2,nothing,batchsize,nx,partial,indices,xsize,nothing,xtype,nothing)
end

function Base.iterate(m::MB,i=0)     # return i+1:i+batchsize
    if i >= m.length || (!m.partial && i + m.batchsize > m.length); return nothing; end
    j=min(i+m.batchsize, m.length)
    ids = m.indices[i+1:j]
    m.xsize[end] = length(ids)
    xbatch = convert(m.xtype, reshape(m.x[:,ids],m.xsize...))
    if m.y == nothing
        return (xbatch,j)
    else
        m.ysize[end] = length(ids)
        ybatch = convert(m.ytype, reshape(m.y[:,ids],m.ysize...))
        return ((xbatch,ybatch),j)
    end
end

function Base.length(m::MB)
    n = m.length / m.batchsize
    m.partial ? ceil(Int,n) : floor(Int,n)
end

function Random.rand(m::MB)
    i = rand(0:(m.length-m.batchsize))
    return next(m, i)[1]
end

"""
    nll(model, data, predict; average=true)

Compute `nll(predict(model,x), y)` for `(x,y)` in `data` and return
the per-instance average (if average=true) or total (if average=false)
negative log likelihood.

"""
function nll(model,data::MB,predict; average=true)
    sum = cnt = 0
    for (x,y) in data
        sum += nll(predict(model,x),y; average=false)
        cnt += length(y)
    end
    average ? sum / cnt : sum
end


"""
    accuracy(model, data, predict; average=true)

Compute `accuracy(predict(model,x), y)` for `(x,y)` in `data` and
return the ratio (if average=true) or the count (if average=false) of
correct answers.

"""
function accuracy(model,data::MB,predict; average=true)
    sum = cnt = 0
    for (x,y) in data
        sum += accuracy(predict(model,x),y; average=false)
        cnt += length(y)
    end
    average ? sum / cnt : sum
end
