for p in ("Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""

This module loads the [MNIST](http://yann.lecun.com/exdb/mnist)
hand-written digits dataset.  There are 60000 training and 10000 test
examples. Each input x consists of 784 pixels representing a 28x28
image.  The pixel values are converted to Float32 and normalized to
[0,1].  Each output y is converted to a ten-dimensional one-hot vector
(a vector that has a single non-zero component) using Float32 elements
indicating the correct class (0-9) for a given image.  10 is used to
represent 0.
```
# Usage:
include(Pkg.dir("Knet/data/mnist.jl"))
using MNIST: xtrn, xtst, ytrn, ytst
# xtrn: 28×28×1×60000 Array{Float32,4}
# xtst: 28×28×1×10000 Array{Float32,4}
# ytrn: 10×60000 Array{Float32,2}
# ytst: 10×10000 Array{Float32,2}
```

"""
module MNIST
using Compat,GZip

function loaddata()
    global xtrn,ytrn,xtst,ytst
    info("Loading MNIST...")
    xtrn = xdata(gzload("train-images-idx3-ubyte.gz"))
    xtst = xdata(gzload("t10k-images-idx3-ubyte.gz"))
    ytrn = ydata(gzload("train-labels-idx1-ubyte.gz"))
    ytst = ydata(gzload("t10k-labels-idx1-ubyte.gz"))
end

function gzload(file; path=Pkg.dir("Knet","data",file), url="http://yann.lecun.com/exdb/mnist/$file")
    if !isfile(path); download(url, path); end
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end

function xdata(a)
    a = view(a,17:length(a))
    reshape(a ./ 255f0, (28,28,1,div(length(a),784)))
end

function ydata(a)
    a = view(a,9:length(a))
    a[a.==0] = 10
    full(sparse(a,1:length(a),1f0,10,length(a)))
end

loaddata()

end # module MNIST
