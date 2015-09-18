# TODO: get rid of the Data abstract type, allow user to use regular arrays as iterables.

import Base: start, next, done

"""
Data is an abstract type for generating data in minibatches.

Its subtypes implement the Iterator interface:

* `start(iter) => state`
* `next(iter,state) => (item,state)`
* `done(iter,state) => Bool`
"""
abstract Data


"""
ItemTensor is a Data subtype that is constructed from a single
array x[d...,i] where the last dimension is interpreted as the
item index.  For non-sequential data.    
"""
type ItemTensor <: Data; x; rng; datasize; epochsize; batchsize; bootstrap; shuffle; batch;
    function ItemTensor(x...; rng=MersenneTwister(), epoch=ccount(x[1]), batch=128, bootstrap=false, shuffle=false)
        nx = ccount(x[1])
        all(xi->ccount(xi)==nx, x) || error("Item count mismatch")
        idx = (shuffle ? shuffle!(rng,[1:nx;]) : nothing)
        buf = map(xi->KUdense(similar(xi, csize(xi,batch))), x)
        new(x, rng, nx, epoch, batch, bootstrap, idx, buf)
    end
end

function next(d::ItemTensor, n)
    idx = nextidx(d,n)
    for i=1:length(d.x)
        cslice!(d.batch[i], d.x[i], idx)
    end
    (d.batch, n+length(idx))
end

# The following can be inherited by other generators:

start(d::Data)=(d.shuffle != nothing && shuffle!(d.rng, d.shuffle); 0)

done(d::Data, n)=(n >= d.epochsize)

function nextidx(d::Data, n)
    nx = d.datasize
    nb = min(d.batchsize, d.epochsize-n)
    if d.bootstrap
        ix = rand(d.rng, 1:nx, nb)
    elseif d.shuffle != nothing
        i1 = mod1(n+1, nx)
        i2 = min(i1+nb-1, nx)
        ix = d.shuffle[i1:i2]
        while length(ix) < nb
            shuffle!(d.rng, d.shuffle)
            i2 = min(nb - length(ix), nx)
            ix = [ix; d.shuffle[1:i2]]
        end
    else
        i1 = mod1(n+1, nx)
        i2 = min(i1+nb-1, nx)
        ix = (i1:i2)
        while length(ix) < nb
            i2 = min(nb - length(ix), nx)
            ix = [ix; 1:i2]
        end
    end
    length(ix) == nb || error()
    return ix
end
