using KUnet

net = [Conv(5,5,1,20), Bias(20), Relu(), Pool(2),
       Conv(5,5,20,50), Bias(50), Relu(), Pool(2),
       Mmul(500,800), Bias(500), Relu(),
       Mmul(10,500), Bias(10), XentLoss()]

require("mnist.jl")
using MNIST: xtrn, ytrn, xtst, ytst
xtrn2 = reshape(xtrn, 28, 28, 1, size(xtrn, 2))
xtst2 = reshape(xtst, 28, 28, 1, size(xtst, 2))

using KUnet: accuracy
setparam!(net; lr=0.1)
for i=1:100
    train(net, xtrn2, ytrn)
    println((i, accuracy(ytst, predict(net, xtst2)), 
                accuracy(ytrn, predict(net, xtrn2))))
end
