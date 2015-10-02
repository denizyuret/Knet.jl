require(Pkg.dir("Knet/test/mnist.jl"))
using Knet
using MNIST: xtrn, ytrn, xtst, ytst

net = [Mmul(64), Bias(), Relu(),
       Mmul(10),  Bias(), XentLoss()]
setparam!(net; lr=0.01)

savenet("net0.jld", net)

@time for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
                accuracy(ytrn, predict(net, xtrn))))
end

net = loadnet("net0.jld")
setparam!(net; lr=0.5)

@time for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
                accuracy(ytrn, predict(net, xtrn))))
end

for h in (128, 256, 512, 1024)
    @show h
    net = [Mmul(h), Bias(), Relu(), Mmul(10),  Bias(), XentLoss()]
    setparam!(net; lr=0.5)
    @time for i=1:100
        train(net, xtrn, ytrn)
        println((i, accuracy(ytst, predict(net, xtst)), 
                 accuracy(ytrn, predict(net, xtrn))))
    end
end

net = [Drop(0.2), Mmul(1024), Bias(), Relu(), 
       Drop(0.5), Mmul(10),  Bias(), XentLoss()]
setparam!(net; lr=0.5)

@time for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
             accuracy(ytrn, predict(net, xtrn))))
end

net = [Drop(0.2), Mmul(4096),  Bias(), Relu(), 
       Drop(0.5), Mmul(4096), Bias(), Relu(), 
       Drop(0.5), Mmul(10),   Bias(), XentLoss()]
setparam!(net; lr=0.5)

@time for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
             accuracy(ytrn, predict(net, xtrn))))
end

net = [Conv(20,5), Bias(), Relu(), Pool(2),
       Conv(50,5), Bias(), Relu(), Pool(2),
       Mmul(500), Bias(), Relu(),
       Mmul(10), Bias(), XentLoss()]
setparam!(net; lr=0.1)

xtrn2 = reshape(xtrn, 28, 28, 1, size(xtrn, 2))
xtst2 = reshape(xtst, 28, 28, 1, size(xtst, 2))

@time for i=1:100
    train(net, xtrn2, ytrn)
    println((i, accuracy(ytst, predict(net, xtst2)), 
             accuracy(ytrn, predict(net, xtrn2))))
end

