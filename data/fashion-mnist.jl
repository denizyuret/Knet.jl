using Pkg; haskey(Pkg.installed(),"CodecZlib") || Pkg.add("CodecZlib")
using CodecZlib

"Where to download fmnist from"
fmnisturl = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"

"Where to download fmnist to"
fmnistdir = joinpath(@__DIR__, "fashion-mnist")

"""

The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
dataset contains images of fashion products(trousers, shirts,
bags...).  There are 60000 training and 10000 test examples. Each
input x consists of 784 pixels representing a 28x28 image. The pixel
values are normalized to [0,1]. Each output y is an integer label
indicating the correct class (1-10) for a given image.  Labels and
descriptions are shown below.

    Label   Description
    1       T-shirt/top
    2       Trouser
    3       Pullover
    4       Dress
    5       Coat
    6       Sandal
    7       Shirt
    8       Sneaker
    9       Bag
    10      Ankle boot

"""

function fmnist()
    global _fmnist_xtrn,_fmnist_ytrn,_fmnist_xtst,_fmnist_ytst,_fmnist_lbls
    if !(@isdefined _fmnist_xtrn)
        @info("Loading FMNIST...")
        _fmnist_xtrn = _fmnist_xdata("train-images-idx3-ubyte.gz")
        _fmnist_xtst = _fmnist_xdata("t10k-images-idx3-ubyte.gz")
        _fmnist_ytrn = _fmnist_ydata("train-labels-idx1-ubyte.gz")
        _fmnist_ytst = _fmnist_ydata("t10k-labels-idx1-ubyte.gz")
        _fmnist_lbls = split("T-shirt/top Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag AnkleBoot")
    end
    return _fmnist_xtrn,_fmnist_ytrn,_fmnist_xtst,_fmnist_ytst,_fmnist_lbls
end

"Utility to view a Fashion MNIST image, requires the Images package"
fmnistview(x,i)=colorview(Gray,permutedims(x[:,:,1,i],(2,1)))

function _fmnist_xdata(file)
    a = _fmnist_gzload(file)[17:end]
    reshape(a ./ 255f0, (28,28,1,div(length(a),784)))
end

function _fmnist_ydata(file)
    _fmnist_gzload(file)[9:end] .+ 0x1
end

function _fmnist_gzload(file)
    if !isdir(fmnistdir)
        mkpath(fmnistdir)
    end
    path = joinpath(fmnistdir,file)
    if !isfile(path)
        url = "$fmnisturl/$file"
        download(url, path)
    end
    f = GzipDecompressorStream(open(path))
    a = read(f)
    close(f)
    return(a)
end

nothing
