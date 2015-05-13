require("mnist.jl")
using KUnet
using MNIST: xtrn, ytrn, xtst, ytst
accuracy(y,z)=mean(findmax(y,1)[2] .== findmax(z,1)[2])
KUnet.gpuseed(1)
srand(1)

net = [Drop(0.2), Mmul(4096, 784), Bias(4096), Relu(), 
       Drop(0.5), Mmul(4096, 4096), Bias(4096), Relu(),
       Drop(0.5), Mmul(10, 4096), Bias(10)
       ]
setparam!(net, :lr, 0.5)

for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
                accuracy(ytrn, predict(net, xtrn))))
end
