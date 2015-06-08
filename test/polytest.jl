include(Pkg.dir("KUnet/test/mnist.jl"))
using KUnet
using MNIST: xtrn, ytrn, xtst, ytst
using KUnet: accuracy

KUnet.atype(Array)
KUnet.ftype(Float32)
w0 = similar(ytrn, size(ytrn,1), 0)
net = Layer[Poly(c=1,d=2,w=w0), PercLoss()]
@time for i=1:10
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
                accuracy(ytrn, predict(net, xtrn))))
end
