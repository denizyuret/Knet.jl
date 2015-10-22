import Base: start, next, done

"""
Data generators should implement the Iterator interface to produce minibatches:

* `start(iter) => state`
* `next(iter,state) => (item,state)`
* `done(iter,state) => Bool`

ItemTensor is a data generator that is constructed from a tuple of
arrays (x[d...,i], y[d...,i],...) where the last dimension is
interpreted as the item index.  Produces tuples of minibatches. For
non-sequential data.

"""
type ItemTensor; data; rng; datasize; epochsize; batchsize; bootstrap; shuffle; batch;
    function ItemTensor(x...; rng=MersenneTwister(), epoch=ccount(x[1]), batch=128, bootstrap=false, shuffle=false)
        nx = ccount(x[1])
        all(xi->ccount(xi)==nx, x) || error("Item count mismatch")
        idx = (shuffle ? shuffle!(rng,[1:nx;]) : nothing)
        buf = map(xi->itembatch(xi,batch), x)
        new(x, rng, nx, epoch, batch, bootstrap, idx, buf)
    end
end

start(d::ItemTensor)=(d.shuffle != nothing && shuffle!(d.rng, d.shuffle); 0)

done(d::ItemTensor, n)=(n + d.batchsize > d.epochsize)

function next(d::ItemTensor, n)
    idx = nextidx(d,n)
    for i=1:length(d.data)
        cslice!(d.batch[i], d.data[i], idx) # 549/556
    end
    (d.batch, n+length(idx))
end

function nextidx(d, n)
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

# DynamicArrayCPU{T}(::Type{T}, d::Dims)=KUdense(Array,T,d)
# DynamicArrayGPU{T}(::Type{T}, d::Dims)=KUdense(CudaArray,T,d)
# SparseArrayCPU{T}(::Type{T}, d::Dims)=spzeros(T,d...)
# SparseArrayGPU{T}(::Type{T}, d::Dims)=CudaSparseMatrixCSR(spzeros(T,d...))

# SparseArrayCPU{T}(::Type{T}, d::Dims)=spzeros(T,d...)
# DynamicArrayCPU{T}(::Type{T}, d::Dims)=KUdense(Array,T,d)
# itembatch(x,n)=(issparse(x)?SparseArrayCPU:DynamicArrayCPU)(eltype(x),csize(x,n))

itembatch(x,n)=(issparse(x) ? spzeros(eltype(x),csize(x,n)...) : Array(eltype(x),csize(x,n)))