using CUDArt
using KUnet
using KUnet: accuracy
require(Pkg.dir("KUnet/test/mnist.jl"))

KUnet.ftype(Float32) # mnist has Float32 data
KUnet.atype(Array)
xtrn = MNIST.xtrn
xtst = MNIST.xtst
ytrn = MNIST.ytrn
ytst = MNIST.ytst
w0 = similar(ytst, size(ytst,1), 0)
d0 = 6
c0 = 1

if true
xnet = Op[Poly(c=c0,d=d0,w=w0), PercLoss()]
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
snet = Op[Poly(c=c0,d=d0,w=w0), PercLoss()]
for i=1:1
    @time train(snet, strn, ytrn; iters=100)
    println((i, size(snet[1].s), 
             accuracy(ytst, predict(snet, stst)),
             )) # accuracy(ytrn, predict(snet, strn))))
end
end # if false

if false # mmul, hcat, ctranspose do not work
KUnet.atype(CudaArray)
cnet = Op[Poly(c=c0,d=d0,w=CudaArray(w0)), PercLoss()]
for i=1:1
    @time train(cnet, xtrn, ytrn; iters=100)
    println((i, size(cnet[1].s), 
             accuracy(ytst, predict(cnet, xtst)),
             )) # accuracy(ytrn, predict(cnet, xtrn))))
end
KUnet.atype(Array)
end # if false
