# Each Layer implements some common functions, stubs are given below.
# forw takes input x and returns output y, possibly setting some state.
# back takes dy, the loss gradient wrt y, calculates loss gradient wrt 
# layer parameters and optionally returns dx, the loss gradient wrt x.
# Some layers overwrite their inputs.

abstract Layer
forw(l::Layer, x; o...)=error("$(typeof(l)) has not implemented forw")
back(l::Layer, dy; o...)=error("$(typeof(l)) has not implemented back")
# copy(l::Layer; o...)=error("$(typeof(l)) has not implemented copy")
update(l::Layer; o...)=nothing
setparam!(l::Layer; o...)=nothing

# LossLayer is slightly different:
# forw only records the outgoing y.
# back takes z, the desired output, and overwrites it with the loss gradient wrt y
# loss takes z, the desired output, and returns a loss value

abstract LossLayer <: Layer
loss(l::LossLayer, z; o...)=error("$(typeof(l)) has not implemented loss")

# Net: Convenience type for an array of layers

typealias Net Array{Layer,1}
forw(n::Net, x; o...)=(for l in n; x=forw(l, x; o...); end; x)
back(n::Net, dy; returndx=false, o...)=(for i=length(n):-1:1; dy=back(n[i],dy; returndx=(i>1||returndx), o...); end; dy)
# copy(n::Net; o...)=Layer[map(l->copy(l; o...),n)...]  # need Layer[] otherwise type may change to e.g. Array{Relu}
update(n::Net; o...)=(for l in n; update(l; o...); end)
setparam!(n::Net; o...)=(for l in n; setparam!(l; o...); end)

# The backprop algorithm

function backprop(net::Net, x, y; o...)
    forw(net, x; o...) # calculate network output given input x
    back(net, y; o...) # calculate derivatives dx,dw given desired output y
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

# Predict implements forw with minibatches.

function predict(net::Net, x, y=nothing; batch=128, o...)
    ninst = size(x, ndims(x))
    (batch == 0 || batch > ninst) && (batch = ninst)
    xx = yy = nothing
    for b = 1:batch:ninst
        e  = min(ninst, b + batch - 1)
        xx = x2b(xx, x, b:e)
        yy = forw(net, xx; predict=true, o...)
        y  = b2y(y, yy, b:e, x)
    end
    return y
end

function b2y(y, b, r, x)
    ys = tuple(size(b)[1:end-1]..., size(x, ndims(x)))
    (y == nothing) && (y = Array(eltype(x), ys))
    @assert size(y) == ys
    @assert eltype(y) == eltype(b)
    yi = 1 + (first(r) - 1) * stride(y, ndims(y))
    copy!(y, yi, b, 1, length(b))
    return y
end

# function b2y_old(y, b, r, x)
#     # The output is always dense
#     n = size(x, ndims(x))
#     ys = tuple(size(b)[1:end-1]..., n)
#     (y == nothing) && (y = (isa(x, AbstractSparseArray) ? Array(eltype(x), ys) : similar(x, ys)))
#     @assert size(y) == ys
#     yi = 1 + (first(r) - 1) * stride(y, ndims(y))
#     copy!(y, yi, b, 1, length(b))
#     return y
# end

function x2b(b, x, r)
    bs = tuple(size(x)[1:end-1]..., length(r))
    if ((b == nothing) || (size(b) != bs))
        b = (gpu()?CudaArray:Array)(eltype(x), bs)
    end
    xi = 1 + (first(r) - 1) * stride(x, ndims(x))
    copy!(b, 1, x, xi, length(b))
    return b
end

function x2b(b, x::SparseMatrixCSC, r)
    # TODO: in-place operation
    # Figure out if b has enough storage
    # Create a new b if not
    # Copy columns to from x to b
    # Copy to gpu if necessary
    b = x[:,r]
    gpu() && (b = gpucopy(b))
    return b
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

using HDF5, JLD

function savenet(filename::String, net::Net)
    GPU && (net = cpucopy(net))
    save(filename, "kunet", net)
end

function loadnet(filename::String)
    net = load(filename, "kunet")
    gpu() ? gpucopy(net) : net
end
