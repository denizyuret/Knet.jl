using Knet
require("mnist.jl")
using MNIST: xtrn, ytrn, xtst, ytst
setseed(42)
net = [Mmul(64), Bias(), Relu(), 
       Mmul(10), Bias(), XentLoss()]
setparam!(net, lr=0.5)
@time for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
                accuracy(ytrn, predict(net, xtrn))))
end
