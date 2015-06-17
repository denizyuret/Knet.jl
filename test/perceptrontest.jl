using KUnet
using KUnet: accuracy
@time require(Pkg.dir("KUnet/test/mnist.jl"))

KUnet.atype(Array)
KUnet.ftype(Float32)
xtrn = float32(255*MNIST.xtrn)
xtst = float32(255*MNIST.xtst)
ytrn = float32(MNIST.ytrn)
ytst = float32(MNIST.ytst)
strn = sparse(xtrn)
stst = sparse(xtst)

if true
xnet = Layer[Perceptron(10)]
@time for i=1:5
    train(xnet, xtrn, ytrn)
    println((i,
             accuracy(ytst, predict(xnet, xtst)),
             accuracy(ytrn, predict(xnet, xtrn))))
end
end # if false

if true
snet = Layer[Perceptron(10)]
@time for i=1:5
    train(snet, strn, ytrn)
    println((i,
             accuracy(ytst, predict(snet, stst)),
             accuracy(ytrn, predict(snet, strn))))
end
end # if false

# This is the stuff that needs to be defined for CudaArrays:
# We probably need gpu kernels for forw, update
using CUDArt
import Base: .+, getindex, ./, -

if false
.+(a::CudaArray{Float32,2}, b::CudaArray{Float32,2})=a
getindex(::CudaArray{Float32,2}, ::Int64, ::Int64)=0
./(a::CudaArray{Float32,2}, ::Int64)=a
-(a::CudaArray{Float32,2}, b::CudaArray{Float32,2})=a

KUnet.atype(CudaArray)
cnet = Layer[Perceptron(10)]
@time for i=1:5
    train(cnet, xtrn, ytrn; iters=100)
    println((i,
             accuracy(ytst, predict(cnet, xtst)),
             accuracy(ytrn, predict(cnet, xtrn))))
end
KUnet.atype(Array)
end # if false

# Similar stuff needs to be defined for sparse cuda arrays.