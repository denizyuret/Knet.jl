using KUnet
include("mnist.jl")
using MNIST: xtrn, ytrn, xtst, ytst
ztrn,ztst = similar(ytrn),similar(ytst)
setseed(42)

info("Testing simple ffnn")
net = Net(Mmul(64), Bias(), Relu(), 
          Mmul(10), Bias(), XentLoss())
setparam!(net, lr=0.5)
@time for i=1:10
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst, ztst)),
                accuracy(ytrn, predict(net, xtrn, ztrn))))
end

info("Testing lenet")
lenet = [Conv(20,5), Bias(), Relu(), Pool(2),
         Conv(50,5), Bias(), Relu(), Pool(2),
         Mmul(500), Bias(), Relu(),
         Mmul(10), Bias(), XentLoss()]

setparam!(lenet; lr=0.1)
xtrn2 = reshape(xtrn, 28, 28, 1, size(xtrn, 2))
xtst2 = reshape(xtst, 28, 28, 1, size(xtst, 2))
@time for i=1:3
    train(lenet, xtrn2, ytrn)
    println((i, accuracy(ytst, predict(lenet, xtst2)), 
             accuracy(ytrn, predict(lenet, xtrn2))))
end


### DEAD CODE

# info("Testing simple ffnn")
# net = [Mmul(64), Bias(), Relu(), 
#        Mmul(10), Bias(), XentLoss()]
# setparam!(net, lr=0.5)
# @time for i=1:10
#     train(net, xtrn, ytrn)
#     println((i, accuracy(ytst, predict(net, xtst)), 
#                 accuracy(ytrn, predict(net, xtrn))))
# end

# info("Testing lenet")
# lenet = [Conv(20,5), Bias(), Relu(), Pool(2),
#          Conv(50,5), Bias(), Relu(), Pool(2),
#          Mmul(500), Bias(), Relu(),
#          Mmul(10), Bias(), XentLoss()]

# setparam!(lenet; lr=0.1)
# xtrn2 = reshape(xtrn, 28, 28, 1, size(xtrn, 2))
# xtst2 = reshape(xtst, 28, 28, 1, size(xtst, 2))
# @time for i=1:3
#     train(lenet, xtrn2, ytrn)
#     println((i, accuracy(ytst, predict(lenet, xtst2)), 
#              accuracy(ytrn, predict(lenet, xtrn2))))
# end
