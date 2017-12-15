for p in ("GZip",)
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using GZip

"Where to download mnist from"
mnisturl = "http://yann.lecun.com/exdb/mnist"

"Where to download mnist to"
mnistdir = Pkg.dir("Knet","data","mnist")

"""

This utility loads the [MNIST](http://yann.lecun.com/exdb/mnist)
hand-written digits dataset.  There are 60000 training and 10000 test
examples. Each input x consists of 784 pixels representing a 28x28
grayscale image.  The pixel values are converted to Float32 and
normalized to [0,1].  Each output y is a UInt8 indicating the correct
class.  10 is used to represent the digit 0.

```
# Usage:
include(Pkg.dir("Knet/data/mnist.jl"))
xtrn, ytrn, xtst, ytst = mnist()
# xtrn: 28×28×1×60000 Array{Float32,4}
# ytrn: 60000-element Array{UInt8,1}
# xtst: 28×28×1×10000 Array{Float32,4}
# ytst: 10000-element Array{UInt8,1}
```

"""
function mnist()
    global _mnist_xtrn,_mnist_ytrn,_mnist_xtst,_mnist_ytst
    if !isdefined(:_mnist_xtrn)
        info("Loading MNIST...")
        _mnist_xtrn = _mnist_xdata("train-images-idx3-ubyte.gz")
        _mnist_xtst = _mnist_xdata("t10k-images-idx3-ubyte.gz")
        _mnist_ytrn = _mnist_ydata("train-labels-idx1-ubyte.gz")
        _mnist_ytst = _mnist_ydata("t10k-labels-idx1-ubyte.gz")
    end
    return _mnist_xtrn,_mnist_ytrn,_mnist_xtst,_mnist_ytst
end

"Utility to view a MNIST image, requires the Images package"
mnistview(x,i)=colorview(Gray,permutedims(x[:,:,1,i],(2,1)))

function _mnist_xdata(file)
    a = _mnist_gzload(file)[17:end]
    reshape(a ./ 255f0, (28,28,1,div(length(a),784)))
end

function _mnist_ydata(file)
    a = _mnist_gzload(file)[9:end]
    a[a.==0] = 10
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
    f = gzopen(path)
    a = read(f)
    close(f)
    return(a)
end

function mnistgrid(y; gridsize=(4,4), scale=2.0, shape=(28,28))
    y = reshape(y, shape..., size(y)[end])
    y = map(x->y[:,:,x]', [1:size(y,3)...])
    shp = map(x->Int(round(x*scale)), shape)
    y = map(x->Images.imresize(x,shp), y)
    gridx, gridy = gridsize
    outdims = (gridx*shp[1]+gridx+1,gridy*shp[2]+gridy+1)
    out = zeros(outdims...)
    for k = 1:gridx+1; out[(k-1)*(shp[1]+1)+1,:] = 1.0; end
    for k = 1:gridy+1; out[:,(k-1)*(shp[2]+1)+1] = 1.0; end

    x0 = y0 = 2
    for k = 1:length(y)
        x1 = x0+shp[1]-1
        y1 = y0+shp[2]-1
        out[x0:x1,y0:y1] = y[k]

        y0 = y1+2
        if k % gridy == 0
            x0 = x1+2
            y0 = 2
        else
            y0 = y1+2
        end
    end

    return convert(Array{Float64,2}, map(x->isnan(x)?0:x, out))
end

nothing
