module MNIST
using GZip
using KUnet
using CUDArt

const mnisturl = "http://yann.lecun.com/exdb/mnist"
const xtrn_file = "train-images-idx3-ubyte.gz"
const ytrn_file = "train-labels-idx1-ubyte.gz"
const xtst_file = "t10k-images-idx3-ubyte.gz"
const ytst_file = "t10k-labels-idx1-ubyte.gz"

type Data xtrn; ytrn; xtst; ytst; end

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
    reshape(a, 28*28, Int(length(a)/(28*28)))
end

xtrn = readimages(xtrn_file)
ytrn = readlabels(ytrn_file)
xtst = readimages(xtst_file)
ytst = readlabels(ytst_file)

function train(net, epochs=10)
    for i=1:epochs
        KUnet.train(net, xtrn, ytrn)
        y = KUnet.predict(net, xtst)
        println((i, mean(findmax(y,1)[2] .== findmax(ytst,1)[2]), length(keys(CUDArt.cuda_ptrs)), KUnet.gpumem()))
    end
end

end
