"Minibatched data"
type MB; x; y; batchsize; length; partial; indices; xsize; ysize; xtype; ytype; end

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

Base.start(m::MB)=0

function Base.done(m::MB,i)
    i >= m.length || (!m.partial && i + m.batchsize > m.length)
end

function Base.next(m::MB,i)     # return i+1:i+batchsize
    j=min(i+m.batchsize, m.length)
    ids = m.indices[i+1:j]
    m.xsize[end] = m.ysize[end] = length(ids)
    xbatch = convert(m.xtype, reshape(m.x[:,ids],m.xsize...))
    ybatch = convert(m.ytype, reshape(m.y[:,ids],m.ysize...))
    return ((xbatch,ybatch),j)
end

function Base.length(m::MB)
    n = m.length / m.batchsize
    m.partial ? ceil(Int,n) : floor(Int,n)
end


"""
    nll(data, model, predict; average=true)

Compute `nll(predict(model,x), y)` for `(x,y)` in `data` and return
the per-instance average (if average=true) or total (if average=false)
negative log likelihood.

"""
function nll(data::MB,model,predict; average=true)
    sum = cnt = 0
    for (x,y) in data
        sum += nll(predict(model,x),y; average=false)
        cnt += length(y)
    end
    average ? sum / cnt : sum
end


"""
    accuracy(data, model, predict; average=true)

Compute `accuracy(predict(model,x), y)` for `(x,y)` in `data` and
return the ratio (if average=true) or the count (if average=false) of
correct answers.

"""
function accuracy(data::MB,model,predict; average=true)
    sum = cnt = 0
    for (x,y) in data
        sum += accuracy(predict(model,x),y; average=false)
        cnt += length(y)
    end
    average ? sum / cnt : sum
end
