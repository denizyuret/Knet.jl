module Mnist
using GZip

const mnisturl = "http://yann.lecun.com/exdb/mnist"
const xtrn_file = "train-images-idx3-ubyte.gz"
const ytrn_file = "train-labels-idx1-ubyte.gz"
const xtst_file = "t10k-images-idx3-ubyte.gz"
const ytst_file = "t10k-labels-idx1-ubyte.gz"
xtrn = ytrn = xtst = ytst = nothing
xtrnS = ytrnS = xtstS = ytstS = nothing
xtrn4 = ytrn4 = xtst4 = ytst4 = nothing

function wgetzcat(gz)
    isfile(gz) || run(`wget $mnisturl/$gz`)
    fh = GZip.open(gz)
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
    xtrn == nothing || return
    xtrn = readimages(xtrn_file)
    ytrn = readlabels(ytrn_file)
    xtst = readimages(xtst_file)
    ytst = readlabels(ytst_file)
end

function conv4dmnist()
    global xtrn4, ytrn4, xtst4, ytst4
    xtrn == nothing && loadmnist()
    xtrnS = sparse(xtrn)
    ytrnS = sparse(ytrn)
    xtstS = sparse(xtst)
    ytstS = sparse(ytst)
end

function sparsemnist()
    global xtrnS, ytrnS, xtstS, ytstS
    xtrn == nothing && loadmnist()
    xtrnS = sparse(xtrn)
    ytrnS = sparse(ytrn)
    xtstS = sparse(xtst)
    ytstS = sparse(ytst)
end

end # module

"""
MNIST returns a pair of data generators (train and test) for the
handwritten digit recognition problem from
http://yann.lecun.com/exdb/mnist.  It takes the same keyword arguments
as ItemTensor.  The data is downloaded if necessary.
"""
function MNIST(; sparse=false, conv4d=true, o...)
    Mnist.xtrn == nothing && (@date Mnist.loadmnist())
    if !sparse
        (ItemTensor(Mnist.xtrn, Mnist.ytrn; o...),
         ItemTensor(Mnist.xtst, Mnist.ytst; o...))
    else
        Mnist.xtrnS == nothing && (@date Mnist.sparsemnist())
        (ItemTensor(Mnist.xtrnS, Mnist.ytrnS; o...),
         ItemTensor(Mnist.xtstS, Mnist.ytstS; o...))
    end
end
