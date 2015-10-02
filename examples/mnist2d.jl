# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.

using Base.Test
using Knet
isdefined(:MNIST) || include("mnist.jl")
include("mlp.jl")

function mnist2d(args=ARGS)
    setseed(42)
    nbatch=100

    dtrn = ItemTensor(MNIST.xtrn, MNIST.ytrn; batch=nbatch)
    dtst = ItemTensor(MNIST.xtst, MNIST.ytst; batch=nbatch)

    x0 = copy(dtrn.data[1])
    y0 = copy(dtrn.data[2])

    info("Testing simple mlp on MNIST")

    prog = mlp(layers=(64,10), loss=softmax, actf=relu, winit=Gaussian(0,.01), binit=Constant(0))
    net = Net(prog)

    setopt!(net, lr=0.5)
    l=w=g=0
    @time for i=1:3
        @show (l,w,g) = train(net, dtrn; gclip=0, gcheck=100, getloss=true, getnorm=true, atol=0.01, rtol=0.001)
        @show (test(net, dtrn), accuracy(net, dtrn))
        @show (test(net, dtst), accuracy(net, dtst))
    end

    @test isequal(x0,dtrn.data[1])
    @test isequal(y0,dtrn.data[2])
    return (l,w,g)
end

!isinteractive() && !isdefined(:load_only) && mnist2d(ARGS)

### SAMPLE RUN

# INFO: Loading MNIST...
#   5.736248 seconds (362.24 k allocations: 502.003 MB, 1.35% gc time)
# INFO: Testing simple mlp
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.001) = (0.37387532f0,18.511799f0,2.8433793f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.21288027f0,0.9327666666666666)
# (test(net,dtst),accuracy(net,dtst)) = (0.2148458f0,0.9289)
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.001) = (0.14995994f0,22.26936f0,3.9932733f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.13567321f0,0.9574)
# (test(net,dtst),accuracy(net,dtst)) = (0.14322147f0,0.9546)
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.001) = (0.10628127f0,24.865437f0,3.5134742f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.100041345f0,0.9681833333333333)
# (test(net,dtst),accuracy(net,dtst)) = (0.114785746f0,0.9641)
#  10.373502 seconds (11.14 M allocations: 557.680 MB, 1.25% gc time)
