module MNIST
using GZip

const mnisturl = "http://yann.lecun.com/exdb/mnist"
const xtrn = "train-images-idx3-ubyte.gz"
const ytrn = "train-labels-idx1-ubyte.gz"
const xtst = "t10k-images-idx3-ubyte.gz"
const ytst = "t10k-labels-idx1-ubyte.gz"

type Data xtrn; ytrn; xtst; ytst; end

function wgetzcat(gz)
    isfile(gz) || run(`wget $mnisturl/$gz`)
    fh = GZip.open(gz)
    a = readbytes(fh)
    close(fh)
    a
end

function readlabels(gz)
    a=wgetzcat(gz)
    n=length(a)
    full(sparse(convert(Vector{Int}, a[9:n]+1), 1:n-8, 1.0f0, 10, n-8))
end

function readimages(gz)
    a=wgetzcat(gz)
    n=length(a)
    r=28*28
    c=Int((n-16)/r)
    reshape(a[17:n],r,c) ./ 255.0f0
end

load()=Data(readimages(xtrn), readlabels(ytrn), readimages(xtst), readlabels(ytst))

end
