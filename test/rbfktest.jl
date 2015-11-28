using CUDArt
using Knet
using Knet: accuracy
require(Pkg.dir("Knet/test/mnist.jl"))

Knet.ftype(Float32) # mnist has Float32 data
Knet.atype(Array)
xtrn = MNIST.xtrn
xtst = MNIST.xtst
ytrn = MNIST.ytrn
ytst = MNIST.ytst
w0 = similar(ytst, size(ytst,1), 0)
g0 = 0.1

if true
xnet = Layer[Rbfk(gamma=g0,w=w0), PercLoss()]
for i=1:1
    @time train(xnet, xtrn, ytrn; iters=100)
    println((i, size(xnet[1].s), 
             accuracy(ytst, predict(xnet, xtst)),
             )) # accuracy(ytrn, predict(xnet, xtrn))))
end
end # if false

if true
strn = sparse(MNIST.xtrn)
stst = sparse(MNIST.xtst)
snet = Layer[Rbfk(gamma=g0,w=w0), PercLoss()]
for i=1:1
    @time train(snet, strn, ytrn; iters=100)
    println((i, size(snet[1].s), 
             accuracy(ytst, predict(snet, stst)),
             )) # accuracy(ytrn, predict(snet, strn))))
end
end # if false

if false # mmul, hcat, ctranspose do not work
Knet.atype(CudaArray)
cnet = Layer[Rbfk(gamma=g0,w=CudaArray(w0)), PercLoss()]
for i=1:1
    @time train(cnet, xtst, ytst; iters=1)
    println((i, size(cnet[1].s), 
             accuracy(ytst, predict(cnet, xtst)),
             )) # accuracy(ytrn, predict(cnet, xtrn))))
end
Knet.atype(Array)
end # if false
