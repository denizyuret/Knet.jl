# Each Layer implements some common functions, stubs are given below.
# forw takes input x and returns output y, possibly setting some state.
# back takes dy, the loss gradient wrt y, calculates loss gradient wrt 
# layer parameters and optionally returns dx, the loss gradient wrt x.
# Some layers overwrite their inputs.

abstract Layer
forw(l::Layer, x; o...)=error("$(typeof(l)) has not implemented forw")
back(l::Layer, dy; o...)=error("$(typeof(l)) has not implemented back")
param(l::Layer)=nothing
update(l::Layer; o...)=update(param(l); o...)
setparam!(l::Layer; o...)=setparam!(param(l); o...)

# Net: Convenience type for an array of layers

typealias Net Array{Layer,1}
forw(n::Net, x; o...)=(for l in n; x=forw(l, x; o...); end; x)
back(n::Net, dy; returndx=false, o...)=(for i=length(n):-1:1; dy=back(n[i],dy; returndx=(i>1||returndx), o...); end; dy)
update(n::Net; o...)=(for l in n; update(l; o...); end; n)
setparam!(n::Net; o...)=(for l in n; setparam!(l; o...); end; n)

# The backprop algorithm

function backprop(net::Net, x, y; o...)
    forw(net, x; o...) # calculate network output given input x
    back(net, y; o...) # calculate derivatives dx,dw given desired output y
end

# Train implements backprop with updates and minibatches.
# It runs for one epoch by default, iters can be specified to stop earlier.

function train(net::Net, x, y; batch=128, shuffle=false, iters=0, o...)
    shuffle && ((x,y)=shufflexy!(x,y))
    ninst = ccount(x)
    ninst==0 && (return warn("No instances"))
    (batch == 0 || batch > ninst) && (batch = ninst)
    (xx,yy) = (xbatch(x, batch), ybatch(y, batch))
    for b = 1:batch:ninst
        e = min(ninst, b + batch - 1)
        (xx,yy) = (cslice!(xx, x, b:e), cslice!(yy, y, b:e))
        backprop(net, xx, yy; o...)
        update(net; o...)
        (iters > 0) && (e/batch >= iters) && break
        gpu() && (gpumem() < (1<<28)) && gc() # need this until julia triggers gc() when gpumem is low
    end
#    strip!(net)
end

# Predict implements forw with minibatches.

function predict(net::Net, x, y=nothing; batch=128, o...)
    ninst = size(x, ndims(x))
    (batch == 0 || batch > ninst) && (batch = ninst)
    (xx,yy) = (xbatch(x, batch), nothing)
    gpu() && (gpumem() < (1<<28)) && gc() # need this until julia triggers gc() when gpumem is low
    for b = 1:batch:ninst
        e  = min(ninst, b + batch - 1)
        xx = cslice!(xx, x, b:e) # 1114
        yy = forw(net, xx; predict=true, o...) # 11587
        (y == nothing) && (y = dsimilar(x, (clength(yy), ccount(x))))
        y = ccopy!(y, b, yy)
    end
    return y
end

accuracy(y,z)=mean(findmax(convert(Array,y),1)[2] .== findmax(convert(Array,z),1)[2])

# Minibatches get created on GPU if gpu() is true:

barray()=(gpu()?CudaArray:Array)

# X batches preserve the sparsity of the input, they use the KU
# versions for resizeability (cslice!).

xbatch(x,b)=(issparse(x) ?
             KUsparse(barray(), eltype(x), itype(x), csize(x,b)) : 
             KUdense(barray(), eltype(x), csize(x,b)))

# Y batches are always dense, because Y should be always dense.  We
# use KUdense for resizeability (cslice!):

ybatch(y,b)=KUdense(barray(), eltype(y), csize(y,b))

# The final prediction output y should match the input x as closely as
# possible except for being dense.

dsimilar(x,d)=(isa(x, SparseMatrixCSC) ? Array(eltype(x), d) :
               isa(x, Sparse{Array}) ? Array(eltype(x), d) :
               isa(x, Sparse{CudaArray}) ? CudaArray(eltype(x), d) :
               isa(x, KUsparse) ? KUdense(atype(x), eltype(x), d) :
               isa(x, KUdense) ? KUdense(atype(x), eltype(x), d) :
               similar(x, d))

dtype(x)=(isa(x, SparseMatrixCSC) ? Array :
          isa(x, Sparse) ? atype(x) :
          isa(x, KUsparse) ? KUdense{atype(x)} :
          isa(x, KUdense) ? KUdense{atype(x)} :
          atype(x))

function dsimilar!(l, n, x, dims)
    if (!isdefined(l,n) || 
        !isa(l.(n), dtype(x)) || 
        (atype(l.(n)) != atype(x)) || 
        (eltype(l.(n)) != eltype(x)))
        l.(n) = dsimilar(x, dims)
    elseif (size(l.(n)) != dims)
        l.(n) = resize!(l.(n), dims)
    end
    return l.(n)
end

function shufflexy!(x,y)
    nx = size(x, ndims(x))
    ny = size(y, ndims(y))
    @assert nx == ny
    r = randperm(nx)
    x = x[map(n->1:n,size(x)[1:end-1])...,r]
    y = y[map(n->1:n,size(y)[1:end-1])...,r]
    return (x,y)
end

function strip!(l::Layer)
    for f in fieldnames(l)
        isdefined(l,f) || continue
        isa(l.(f), KUparam) && strip!(l.(f))
        in(f, (:x, :x2, :y, :dx, :dy, :xdrop)) && (l.(f)=nothing)
    end
    return l
end

strip!(p::KUparam)=(p.init=p.diff=nothing;p)
strip!(n::Net)=(for l in n; strip!(l); end; gc(); n)

using HDF5, JLD

function savenet(filename::String, net::Net)
    net = strip!(net)
    GPU && (net = cpucopy(net))
    save(filename, "kunet", net)
end

function loadnet(filename::String)
    net = load(filename, "kunet")
    net = strip!(net)
    gpu() ? gpucopy(net) : net
end

