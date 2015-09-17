module MNISTmod
using GZip
using KUnet
export MNIST

MNISTDATA=nothing

"""
MNIST returns a pair of data generators (train and test) for the
handwritten digit recognition problem from
http://yann.lecun.com/exdb/mnist.  It takes the same keyword arguments
as ItemTensor.  The data is downloaded if necessary.
"""
function MNIST(;o...)
    global MNISTDATA
    MNISTDATA==nothing && (@date MNISTDATA=LoadMNIST())
    (ItemTensor(MNISTDATA.xtrn, MNISTDATA.ytrn; o...),
     ItemTensor(MNISTDATA.xtst, MNISTDATA.ytst; o...))
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

end # module