using Pkg; haskey(Pkg.installed(),"CodecZlib") || Pkg.add("CodecZlib")
using CodecZlib

"Where to download mnist from"
mnisturl = "http://yann.lecun.com/exdb/mnist"

"Where to download mnist to"
mnistdir = joinpath(@__DIR__, "mnist")

"""

This utility loads the [MNIST](http://yann.lecun.com/exdb/mnist)
hand-written digits dataset.  There are 60000 training and 10000 test
examples. Each input x consists of 784 pixels representing a 28x28
grayscale image.  The pixel values are converted to Float32 and
normalized to [0,1].  Each output y is a UInt8 indicating the correct
class.  10 is used to represent the digit 0.

```
# Usage:
include("mnist.jl")
xtrn, ytrn, xtst, ytst = mnist()
# xtrn: 28×28×1×60000 Array{Float32,4}
# ytrn: 60000-element Array{UInt8,1}
# xtst: 28×28×1×10000 Array{Float32,4}
# ytst: 10000-element Array{UInt8,1}
```

"""
function mnist()
    global _mnist_xtrn,_mnist_ytrn,_mnist_xtst,_mnist_ytst
    if !@isdefined(_mnist_xtrn)
        @info("Loading MNIST...")
        _mnist_xtrn = _mnist_xdata("train-images-idx3-ubyte.gz")
        _mnist_xtst = _mnist_xdata("t10k-images-idx3-ubyte.gz")
        _mnist_ytrn = _mnist_ydata("train-labels-idx1-ubyte.gz")
        _mnist_ytst = _mnist_ydata("t10k-labels-idx1-ubyte.gz")
    end
    return _mnist_xtrn,_mnist_ytrn,_mnist_xtst,_mnist_ytst
end

"Utility to view a MNIST image, requires Images."
mnistview(x,i)=colorview(Gray,permutedims(x[:,:,1,i],(2,1)))

"Return minibatched mnist data, requires Knet."
function mnistdata(;batchsize=100,xtype=(Knet.gpu()>=0 ? Knet.KnetArray{Float32} : Array{Float32}), o...)
    if !@isdefined(_mnist_xtrn); mnist(); end
    Knet.minibatch(_mnist_xtrn, _mnist_ytrn, batchsize; xtype=xtype, o...),
    Knet.minibatch(_mnist_xtst, _mnist_ytst, batchsize; xtype=xtype, o...)
end

function _mnist_xdata(file)
    a = _mnist_gzload(file)[17:end]
    reshape(a ./ 255f0, (28,28,1,div(length(a),784)))
end

function _mnist_ydata(file)
    a = _mnist_gzload(file)[9:end]
    a[a.==0] .= 10
    # full(sparse(a,1:length(a),1f0,10,length(a)))
    return a
end

function _mnist_gzload(file)
    if !isdir(mnistdir)
        mkpath(mnistdir)
    end
    path = joinpath(mnistdir,file)
    if !isfile(path)
        url = "$mnisturl/$file"
        download(url, path)
    end
    f = GzipDecompressorStream(open(path))
    a = read(f)
    close(f)
    return(a)
end

nothing
