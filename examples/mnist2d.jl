# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.

using Base.Test
using KUnet
using KUnet: params
isdefined(:MNIST) || include("mnist.jl")
setseed(42)
nbatch=100

dtrn = ItemTensor(MNIST.xtrn, MNIST.ytrn; batch=nbatch)
dtst = ItemTensor(MNIST.xtst, MNIST.ytst; batch=nbatch)

x0 = copy(dtrn.data[1])            # TODO: should rename x a more descriptive data
y0 = copy(dtrn.data[2])

info("Testing simple mlp")

macro net(a) a end
type input <: Op; end
type par <: Op; end
type dot <: Op; end
type add <: Op; end
type relu <: Op; end
type soft <: Op; end

# Net(Mmul(64), Bias(), Relu(), Mmul(10), Bias(), Soft(), SoftLoss())

axpb(;n=1) = quote
    x1 = input()
    w1 = par($n,0)
    x2 = dot(w1,x1)
    b2 = par(0)
    x3 = add(b2,x2)
end

net = quote
    x1 = input()
    x2 = axpb(x1; n=64)
    x3 = relu(x2)
    x4 = axpb(x3; n=10)
    x5 = soft(x4)
end

# setparam!(net, lr=0.5)
# @time for i=1:3
#     @show (l,w,g) = train(net, dtrn; gclip=0, gcheck=100, getloss=true, getnorm=true, atol=0.01, rtol=0.01)
#     @show (test(net, dtrn), accuracy(net, dtrn))
#     @show (test(net, dtst), accuracy(net, dtst))
# end

# @test isequal(x0,dtrn.data[1])
# @test isequal(y0,dtrn.data[2])
