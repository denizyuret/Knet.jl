module MNIST
using GZip

const mnisturl = "http://yann.lecun.com/exdb/mnist"
const xtrn_file = "train-images-idx3-ubyte.gz"
const ytrn_file = "train-labels-idx1-ubyte.gz"
const xtst_file = "t10k-images-idx3-ubyte.gz"
const ytst_file = "t10k-labels-idx1-ubyte.gz"

function wgetzcat(gz; gzpath=joinpath(dirname(@__FILE__),"..","data",gz))
    if !isfile(gzpath)
        info("Downloading $mnisturl/$gz to $gzpath")
        download("$mnisturl/$gz", gzpath)
    end
    fh = GZip.open(gzpath)
    a = readbytes(fh)
    close(fh)
    a
end

function readlabels(gz)
    a=convert(Vector{Int}, wgetzcat(gz)[9:end])
    a[a.==0]=10
    full(sparse(a, 1:length(a), 1.0f0))
end

function readimages(gz)
    a=(wgetzcat(gz)[17:end] ./ 255.0f0)
    reshape(a, 28, 28, 1, div(length(a),(28*28)))
end

function loadmnist()
    global xtrn, ytrn, xtst, ytst
    xtrn = readimages(xtrn_file)
    ytrn = readlabels(ytrn_file)
    xtst = readimages(xtst_file)
    ytst = readlabels(ytst_file)
end

info("Loading MNIST...")
loadmnist()

end # module
