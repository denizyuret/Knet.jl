# Each Layer implements some common functions, stubs are given below.
# forw takes input x and returns output y, possibly setting some state
# back takes dy, the loss gradient wrt y, and returns dx, the loss gradient wrt x
# Some layers overwrite their inputs

abstract Layer
forw(l::Layer, x; o...)=nothing
back(l::Layer, dy; o...)=nothing
copy(l::Layer; o...)=nothing
update(l::Layer; o...)=nothing
setparam!(l::Layer; o...)=nothing

# LossLayer is slightly different:
# forw only records the outgoing y.
# back takes z, the desired output, and overwrites it with the loss gradient wrt y
# loss takes z, the desired output, and returns a loss value

abstract LossLayer <: Layer
loss(l::LossLayer, z; o...)=nothing

# Net: Convenience type for an array of layers

typealias Net Array{Layer,1}
forw(n::Net, x; o...)=(for l in n; x=forw(l, x; o...) end; x)
back(n::Net, dy; o...)=(for i=length(n):-1:1 dy=back(n[i],dy; dx=(i>1), o...) end)
copy(n::Net; o...)=Layer[map(l->copy(l; o...),n)...]  # need Layer[] otherwise type may change to e.g. Array{Relu}
update(n::Net; o...)=(for l in n; update(l; o...); end)
setparam!(n::Net; o...)=(for l in n; setparam!(l; o...); end)

# The backprop algorithm

function backprop(net::Net, x, dy; o...)
    y = forw(net, x; o...) 	# calculate network output y given input x
    back(net, dy; o...)         # calculate derivatives dx,dw given desired output dy
end

# Predict implements forw with minibatches.

function predict(net::Net, x, y=nothing; batch=0, o...)
    ninst = size(x, ndims(x))
    (batch == 0 || batch > ninst) && (batch = ninst)
    xx = yy = y = nothing
    for b = 1:batch:ninst
        e  = min(ninst, b + batch - 1)
        xx = x2b(xx, x, b:e)
        yy = forw(net, xx; fx=false, o...)
        y  = b2y(y, yy, b:e, x)
    end
    return y
end

# Train implements backprop with updates and minibatches.
# It runs for one epoch by default, iters can be specified to stop earlier.

function train(net::Net, x, y; batch=128, shuffle=false, iters=0, o...)
    shuffle && shufflexy!(x,y)
    ninst = size(x, ndims(x))
    (batch == 0 || batch > ninst) && (batch = ninst)
    xx = yy = nothing
    for b = 1:batch:ninst
        e = min(ninst, b + batch - 1)
        xx = x2b(xx, x, b:e)
        yy = x2b(yy, y, b:e)
        backprop(net, xx, yy; o...)
        update(net; o...)
        (iters > 0) && (e/batch >= iters) && break
    end
end

function x2b(b, x, r)
    bs = tuple(size(x)[1:end-1]..., length(r))
    if ((b == nothing) || (size(b) != bs))
        b = Atype(Ftype, bs)
    end
    xi = 1 + (first(r) - 1) * stride(x, ndims(x))
    copy!(b, 1, x, xi, length(b))
end

function b2y(y, b, r, x)
    n = size(x, ndims(x))
    ys = tuple(size(b)[1:end-1]..., n)
    (y == nothing) && (y = similar(x, ys))
    @assert size(y) == ys
    yi = 1 + (first(r) - 1) * stride(y, ndims(y))
    copy!(y, yi, b, 1, length(b))
end

function shufflexy!(x, y)
    xrows,xcols = size2(x)
    yrows,ycols = size2(y)
    @assert xcols == ycols
    x1 = Array(eltype(x), xrows)
    y1 = Array(eltype(y), yrows)
    for n = xcols:-1:2
        r = rand(1:n)
        r == n && continue
        nx = (n-1)*xrows+1; ny = (n-1)*yrows+1
        rx = (r-1)*xrows+1; ry = (r-1)*yrows+1
        copy!(x1, 1, x, nx, xrows)
        copy!(y1, 1, y, ny, yrows)
        copy!(x, nx, x, rx, xrows)
        copy!(y, ny, y, ry, yrows)
        copy!(x, rx, x1, 1, xrows)
        copy!(y, ry, y1, 1, yrows)
    end
end

