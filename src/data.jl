"""
Data is an abstract type for generating data in minibatches.

Its subtypes implement the Iterator interface:

* `start(iter) => state`
* `next(iter,state) => (item,state)`
* `done(iter,state) => Bool`

Implemented subtypes:

* `ItemTensor`
* `TrainMNIST`,`TestMNIST`
* `AddingData`
"""
abstract Data

import Base: start, next, done
using GZip

"""
ItemTensor is a Data subtype that is constructed from a single
array x[d...,i] where the last dimension is interpreted as the
item index.  For non-sequential data.    
"""
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


"""

TrainMNIST and TestMNIST are the data generators for the handwritten
digit recognition problem from http://yann.lecun.com/exdb/mnist.
The constructors take the same options as ItemTensor.

"""

MNIST=nothing

function TrainMNIST(;o...)
    global MNIST
    MNIST==nothing && (@date MNIST=LoadMNIST())
    ItemTensor(MNIST.xtrn, MNIST.ytrn; o...)
end

function TestMNIST(;o...)
    global MNIST
    MNIST==nothing && (@date MNIST=LoadMNIST())
    ItemTensor(MNIST.xtst, MNIST.ytst; o...)
end

type LoadMNIST; xtrn; ytrn; xtst; ytst;
    LoadMNIST()=new(mnist_images(mnist_xtrn),
                    mnist_labels(mnist_ytrn),
                    mnist_images(mnist_xtst),
                    mnist_labels(mnist_ytst))
end

function mnist_images(gz)
    a=(mnist_get(gz)[17:end] ./ 255.0f0)
    reshape(a, 28, 28, 1, div(length(a),28*28))
end

function mnist_labels(gz)
    a=convert(Vector{Int}, mnist_get(gz)[9:end])
    a[a.==0]=10
    full(sparse(a, 1:length(a), 1.0f0))
end

function mnist_get(gz)
    isfile(gz) || run(`wget $mnist_url/$gz`)
    fh = GZip.open(gz)
    a = readbytes(fh)
    close(fh)
    a
end

const mnist_url  = "http://yann.lecun.com/exdb/mnist"
const mnist_xtrn = "train-images-idx3-ubyte.gz"
const mnist_ytrn = "train-labels-idx1-ubyte.gz"
const mnist_xtst = "t10k-images-idx3-ubyte.gz"
const mnist_ytst = "t10k-labels-idx1-ubyte.gz"


"""
This is the data generator for the adding problem from: Le, Q. V.,
Jaitly, N., & Hinton, G. E. (2015). A Simple Way to Initialize
Recurrent Networks of Rectified Linear Units. arXiv preprint
arXiv:1504.00941.
"""
type AddingData <: Data; len; batchsize; epochsize; rng;
    AddingData(len, batchsize, epochsize; rng=MersenneTwister())=new(len, batchsize, epochsize, rng)
end

start(a::AddingData)=0

done(a::AddingData,n)=(n >= a.epochsize)

function next(a::AddingData, n)
    nb = min(a.batchsize, a.epochsize-n)
    x = [ vcat(rand(a.rng,Float32,1,nb),zeros(Float32,1,nb)) for t=1:a.len ]
    y = Array(Float32,1,nb)
    t1 = rand(a.rng,1:a.len,nb)
    t2 = rand(a.rng,1:a.len,nb)
    for b=1:nb
        while t2[b]==t1[b]
            t2[b]=rand(a.rng,1:a.len)
        end
        x[t1[b]][2,b]=1
        x[t2[b]][2,b]=1
        y[b] = x[t1[b]][1,b] + x[t2[b]][1,b]
    end
    return ((x,y), n+nb)
end

