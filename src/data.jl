"""
Data is an abstract type for generating data in minibatches.
Its subtypes should implement the Iterator interface:

* `start(iter) => state`
* `next(iter,state) => (item,state)`
* `done(iter,state) => Bool`
"""
abstract Data

import Base: start, next, done

type ItemTensor <: Data; x; rng; epochsize; batchsize; bootstrap; shuffle; batch;
    function ItemTensor(x...; rng=MersenneTwister(), epoch=ccount(x[1]), batch=128, bootstrap=false, shuffle=false)
        nx = ccount(x[1])
        all(xi->ccount(xi)==nx, x) || error("Item count mismatch")
        idx = (shuffle ? shuffle!(rng,[1:nx;]) : nothing)
        buf = map(xi->KUdense(similar(xi, csize(xi,batch))), x)
        new(x, rng, epoch, batch, bootstrap, idx, buf)
    end
end

start(d::ItemTensor)=(d.shuffle != nothing && shuffle!(d.rng, d.shuffle); 0)

done(d::ItemTensor, n)=(n >= d.epochsize)

function next(d::ItemTensor, n)
    nx = ccount(d.x[1])
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
    for i=1:length(d.x)
        cslice!(d.batch[i], d.x[i], ix)
    end
    (d.batch, n+nb)
end
