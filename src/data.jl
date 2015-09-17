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
    function ItemTensor(x...; rng=MersenneTwister(), epochsize=ccount(x[1]), batchsize=128, bootstrap=false, shuffle=false)
        nx = ccount(x[1])
        all(xi->ccount(xi)==nx, x) || error("Item count mismatch")
        shuffle = (shuffle ? (1:nx) : nothing)
        batch = map(xi->KUdense(cget(xi,1:batchsize)), x)
        new(x, rng, epochsize, batchsize, bootstrap, shuffle, batch)
    end
end

function start(d::ItemTensor)
    if d.shuffle != nothing
        d.shuffle=randperm(d.rng,length(d.shuffle))
    end
    return 0
end

function done(d::ItemTensor, n)
    n >= d.epochsize
end

function next(d::ItemTensor, n)
    nx = ccount(d.x[1])
    nb = min(d.batchsize, d.epochsize-n)
    if d.bootstrap
        ix = rand(d.rng, 1:nx, nb)
    else
        i1 = mod1(n+1, nx)
        i2 = mod1(n+nb, nx)
        if d.shuffle == nothing
            ix = (i1 < i2 ? (i1:i2) : [i1:nx; 1:i2])
            # nb>nx?
        else
            error("This is not working yet")
            # i1==1?
        end
    end
    length(ix) == nb || error()
    for i=1:length(d.x)
        cslice!(d.batch[i], d.x[i], ix)
    end
    (d.batch, n+nb)
end
