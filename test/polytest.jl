using KUnet
using KUnet: accuracy
include(Pkg.dir("KUnet/test/mnist.jl"))
KUnet.ftype(Float32) # mnist has Float32 data
KUnet.atype(Array)

if true
xtrn = MNIST.xtrn
xtst = MNIST.xtst
ytrn = MNIST.ytrn
ytst = MNIST.ytst
w0 = similar(ytrn, size(ytrn,1), 0)
net = Layer[Poly(c=1f0,d=3f0,w=w0), PercLoss()]
for i=1:1
    @time train(net, xtst, ytst)
    println((i, size(net[1].s), 
             accuracy(ytst, predict(net, xtst)),
             )) # accuracy(ytrn, predict(net, xtrn))))
end
end # if false

if true
# KUnet.atype(SparseMatrixCSC)
# Base.SparseMatrixCSC{T}(::Type{T}, d::(Int64,Int64))=spzeros(T,d...)
xtrn = sparse(MNIST.xtrn)
xtst = sparse(MNIST.xtst)
ytrn = (MNIST.ytrn)
ytst = (MNIST.ytst)
w0 = similar(ytrn, size(ytrn,1), 0)
net = Layer[Poly(c=1f0,d=3f0,w=w0), PercLoss()]
for i=1:1
    @time train(net, xtst, ytst)
    println((i, size(net[1].s), 
             accuracy(ytst, predict(net, xtst)),
             )) # accuracy(ytrn, predict(net, xtrn))))
end
end # if false