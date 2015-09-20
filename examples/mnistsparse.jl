# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# Testing sparse arrays.

using Base.Test
using KUnet
using KUnet: params
isdefined(:MNIST) || include("mnist.jl")
setseed(42)
nbatch=100

dtrn = ItemTensor(sparse(MNIST.xtrn), sparse(MNIST.ytrn); batch=nbatch)
dtst = ItemTensor(sparse(MNIST.xtst), sparse(MNIST.ytst); batch=nbatch)

x0 = copy(dtrn.data[1])            # TODO: should rename x a more descriptive data
y0 = copy(dtrn.data[2])

info("Testing simple mlp")
net = Net(Mmul(64), Bias(), Relu(), Mmul(10), Bias(), XentLoss())
setparam!(net, lr=0.5)

test(net, dtst)                 # to init weights
mlp = deepcopy(net.op)

@time for i=1:3
    @show (l,w,g) = train(net, dtrn; gclip=0, gcheck=100, getloss=true, getnorm=true)
    @show (test(net, dtrn), accuracy(net, dtrn))
    @show (test(net, dtst), accuracy(net, dtst))
    train(mlp, dtrn.data[1], dtrn.data[2]; batch=nbatch)
    @test all(map(isequal, params(net), params(mlp)))
    println((i, accuracy(dtst.data[2], predict(mlp, dtst.data[1])),
                accuracy(dtrn.data[2], predict(mlp, dtrn.data[1]))))
end

@test isequal(x0,dtrn.data[1])
@test isequal(y0,dtrn.data[2])

