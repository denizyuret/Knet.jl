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

if true
xnet = Layer[Poly(c=1f0,d=3f0,w=w0), PercLoss()]
for i=1:1
    @time train(xnet, xtrn, ytrn)
    println((i, size(xnet[1].s), 
             accuracy(ytst, predict(xnet, xtst)),
             accuracy(ytrn, predict(xnet, xtrn))))
end
end # if false

if true
strn = sparse(MNIST.xtrn)
stst = sparse(MNIST.xtst)
snet = Layer[Poly(c=1f0,d=3f0,w=w0), PercLoss()]
for i=1:1
    @time train(snet, strn, ytrn)
    println((i, size(snet[1].s), 
             accuracy(ytst, predict(snet, stst)),
             accuracy(ytrn, predict(snet, strn))))
end
end # if false
