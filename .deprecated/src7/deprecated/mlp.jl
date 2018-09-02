# MLP: Convenience type for an array of layers
# This is deprecated, please use the new Net().

typealias MLP Vector{Op}

forw(n::MLP, x; o...)=(for l in n; x=forw(l, x; o...); end; x)
back(n::MLP, dy; returndx=false, o...)=(for i=length(n):-1:1; dy=back(n[i],dy; returndx=(i>1||returndx), o...); end; dy)
loss(n::MLP, dy; y=n[end].y)=loss(n[end], dy; y=y)
update!(n::MLP; o...)=(for l in n; update!(l; o...); end)

function params(r::MLP)
    p = Any[]
    for o in r
        append!(p, params(o))
    end
    return p
end

# All of this should be inherited from Model() and/or implemented with the new Net():
# TODO: batching needs to be done outside?
# TODO: predict with y=nothing
# TODO: train with shuffle, iters
# TODO: strip, savenet, loadnet implement for the new Net().
# TODO: accuracy needs to go somewhere

# The backprop algorithm

function backprop(net::MLP, x, y; o...)
    forw(net, x; o...) # calculate network output given input x
    back(net, y; o...) # calculate derivatives dx,dw given desired output y
end

# Predict implements forw with minibatches.

function predict(net::MLP, x; y=nothing, batch=128, o...)
    ninst = size(x, ndims(x))
    (batch == 0 || batch > ninst) && (batch = ninst)
    (xx,yy) = (xbatch(x, batch), nothing)
    gpu() && (gpumem() < (1<<28)) && gc() # need this until julia triggers gc() when gpumem is low
    for b = 1:batch:ninst
        e  = min(ninst, b + batch - 1)
        xx = cslice!(xx, x, b:e) # 1114
        yy = forw(net, xx; train=false, o...) # 11587
        (y == nothing) && (y = dsimilar(x, (clength(yy), ccount(x))))
        y = ccopy!(y, b, yy)
    end
    return y
end

# Train implements backprop with updates and minibatches.
# It runs for one epoch by default, iters can be specified to stop earlier.

function train(net::MLP, x, y; batch=128, shuffle=false, iters=0, o...)
    shuffle && ((x,y)=shufflexy!(x,y))
    ninst = ccount(x)
    ninst==0 && (return warn("No instances"))
    (batch == 0 || batch > ninst) && (batch = ninst)
    (xx,yy) = (xbatch(x, batch), ybatch(y, batch))
    for b = 1:batch:ninst
        e = min(ninst, b + batch - 1)
        (xx,yy) = (cslice!(xx, x, b:e), cslice!(yy, y, b:e))
        backprop(net, xx, yy; o...)
        update!(net; o...)
        (iters > 0) && (e/batch >= iters) && break
        gpu() && (gpumem() < (1<<28)) && gc() # need this until julia triggers gc() when gpumem is low
    end
    # strip!(net) # this is only for one epoch, why do I strip here?
end

# To shuffle data before each epoch:

function shufflexy!(x,y)
    nx = size(x, ndims(x))
    ny = size(y, ndims(y))
    @assert nx == ny
    r = randperm(nx)
    x = x[map(n->1:n,size(x)[1:end-1])...,r]
    y = y[map(n->1:n,size(y)[1:end-1])...,r]
    return (x,y)
end

# X batches preserve the sparsity of the input, they use the KU
# versions for resizeability (cslice!).

xbatch(x,b)=(issparse(x) ?
             (gpu() ? CudaSparseMatrixCSC(spzeros(eltype(x), size(x)...)) : 
              spzeros(eltype(x), size(x)...)) :
             KUdense(barray(), eltype(x), csize(x,b)))

# Y batches are always dense, because Y should be always dense.  We
# use KUdense for resizeability (cslice!):

ybatch(y,b)=KUdense(barray(), eltype(y), csize(y,b))

# Minibatches get created on GPU if gpu() is true:

barray()=(gpu()?CudaArray:Array)

function strip!(l::Op)
    for f in fieldnames(l)
        isdefined(l,f) || continue
        isa(l.(f), KUparam) && strip!(l.(f))
        in(f, (:x, :x2, :y, :dx, :dy, :xdrop)) && (l.(f)=nothing)
    end
    return l
end

strip!(p::KUparam)=(p.init=p.diff=nothing;p)
strip!(n::MLP)=(for l in n; strip!(l); end; gc(); n)

using JLD

function savenet(filename::String, net::MLP)
    net = strip!(net)
    GPU && (net = cpucopy(net))
    save(filename, "kunet", net)
end

function loadnet(filename::String)
    net = load(filename, "kunet")
    net = strip!(net)
    gpu() ? gpucopy(net) : net
end

accuracy(y,z)=mean(findmax(convert(Array,y),1)[2] .== findmax(convert(Array,z),1)[2])

