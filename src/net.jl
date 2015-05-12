# Each Layer implements three functions, here are the stubs:

abstract Layer
forw(l::Layer, x; o...)=x
back(l::Layer, dy; o...)=dy
update(l::Layer)=nothing

# Net: Convenience type for an array of layers

typealias Net Array{Layer,1}
forw(n::Net, x; fx=true)=(for l in n; x=forw(l, x; fx=fx) end; x)
back(n::Net, dy)=(for i=length(n):-1:1 dy=back(n[i],dy; dx=(i>1)) end)
update(n::Net)=(for l in n; update(l); end)

# The backprop algorithm

function backprop(net::Net, x, dy, loss=softmaxloss)
    y = forw(net, x) 	# y: network output
    loss(y, dy)         # dy: desired output -> gradient
    back(net, dy)       # calculate derivatives
end

# Predict implements forw with minibatches.

function predict(net::Net, x, y=nothing; batch=0)
    ninst = size(x, ndims(x))
    (batch == 0 || batch > ninst) && (batch = ninst)
    xx = yy = y = nothing
    for b = 1:batch:ninst
        e  = min(ninst, b + batch - 1)
        xx = x2b(xx, x, b:e)
        yy = forw(net, xx; fx=false)
        y  = b2y(y, yy, b:e, ninst)
    end
    free(xx)
    return y
end

function x2b(b, x, r)
    bs = tuple(size(x)[1:end-1]..., length(r))
    if ((b == nothing) || (size(b) != bs))
        b == nothing || free(b)
        b = (usegpu ? CudaArray : Array)(eltype(x), bs)
    end
    bi = map(d->1:d, bs)
    xi = tuple(bi[1:end-1]..., r)
    copy!(b, bi, x, xi)
end

function b2y(y, b, r, n)
    ys = tuple(size(b)[1:end-1]..., n)
    (y == nothing) && (y = Array(eltype(b), ys))
    @assert size(y) == ys
    bi = map(d->1:d, size(b))
    yi = tuple(bi[1:end-1]..., r)
    copy!(y, yi, b, bi)
end

# Just a convenience type for training etc.
type XY x; y; XY()=new(); end

function train(net::Net, x, y; batch=128, iters=0, loss=softmaxloss, shuffle=false)
    shuffle && shufflexy!(x,y)
    xrows,xcols = size(x)
    yrows,ycols = size(y)
    (batch == 0 || batch > xcols) && (batch = xcols)
    buf = inittrain(net, x, y, batch)
    for b = 1:batch:xcols
        e = b + batch - 1
        if (e > xcols)
            e = xcols
            chksize(buf, :x, net[1].w, (xrows, e-b+1))
            chksize(buf, :y, net[end].w, (yrows, e-b+1))
        end
        copy!(buf.x, (1:xrows,1:e-b+1), x, (1:xrows,b:e))
        copy!(buf.y, (1:yrows,1:e-b+1), y, (1:yrows,b:e))
        backprop(net, buf.x, buf.y, loss)
        for l in net
            isdefined(l,:w) && update(l.w, l.dw, l.pw)
            isdefined(l,:b) && update(l.b, l.db, l.pb)
        end
        iters > 0 && e/batch >= iters && break
    end
    free(buf.x); free(buf.y) # this should not be necessary now that gc() works...
end

function inittrain(net::Net, x, y, batch)
    for l in net
        isdefined(l,:w) && !isdefined(l,:pw) && (l.pw = UpdateParam())    
        isdefined(l,:b) && !isdefined(l,:pb) && (l.pb = UpdateParam())
    end
    buf = XY()
    chksize(buf, :x, net[1].w, (size(x, 1), batch))
    chksize(buf, :y, net[end].w, (size(y, 1), batch))
    return buf
end

