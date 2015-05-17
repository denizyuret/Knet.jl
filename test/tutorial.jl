include(Pkg.dir("KUnet/test/mnist.jl"))
using KUnet
using MNIST: xtrn, ytrn, xtst, ytst
using KUnet: accuracy

if false
end # if false

net = [Mmul(64,784), Bias(64), Relu(),
       Mmul(10,64),  Bias(10), XentLoss()]
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
    net = [Mmul(h,784), Bias(h), Relu(), Mmul(10,h),  Bias(10), XentLoss()]
    setparam!(net; lr=0.5)
    @time for i=1:100
        train(net, xtrn, ytrn)
        println((i, accuracy(ytst, predict(net, xtst)), 
                 accuracy(ytrn, predict(net, xtrn))))
    end
end

net = [Drop(0.2), Mmul(1024,784), Bias(1024), Relu(), 
       Drop(0.5), Mmul(10,1024),  Bias(10), XentLoss()]
setparam!(net; lr=0.5)

@time for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
             accuracy(ytrn, predict(net, xtrn))))
end

net = [Drop(0.2), Mmul(4096,784),  Bias(4096), Relu(), 
       Drop(0.5), Mmul(4096,4096), Bias(4096), Relu(), 
       Drop(0.5), Mmul(10,4096),   Bias(10), XentLoss()]
setparam!(net; lr=0.5)

@time for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
             accuracy(ytrn, predict(net, xtrn))))
end

net = [Conv(5,5,1,20), Bias(20), Relu(), Pool(2),
       Conv(5,5,20,50), Bias(50), Relu(), Pool(2),
       Mmul(500,800), Bias(500), Relu(),
       Mmul(10,500), Bias(10), XentLoss()]
setparam!(net; lr=0.1)

xtrn2 = reshape(xtrn, 28, 28, 1, size(xtrn, 2))
xtst2 = reshape(xtst, 28, 28, 1, size(xtst, 2))

@time for i=1:100
    train(net, xtrn2, ytrn)
    println((i, accuracy(ytst, predict(net, xtst2)), 
             accuracy(ytrn, predict(net, xtrn2))))
end

