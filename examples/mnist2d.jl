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
net = Net(Mmul(64), Bias(), Relu(), Mmul(10), Bias(), Soft(), SoftLoss())
setparam!(net, lr=0.5)

test(net, dtst)                 # to init weights
mlp = deepcopy(net.op)

@time for i=1:3
    @show (l,w,g) = train(net, dtrn; gclip=0, gcheck=100, getloss=true, getnorm=true, atol=0.01, rtol=0.01)
    @show (test(net, dtrn), accuracy(net, dtrn))
    @show (test(net, dtst), accuracy(net, dtst))
    train(mlp, dtrn.data[1], dtrn.data[2]; batch=nbatch)
    @test all(map(isequal, params(net), params(mlp)))
    println((i, accuracy(dtst.data[2], predict(mlp, dtst.data[1])),
                accuracy(dtrn.data[2], predict(mlp, dtrn.data[1]))))
end

@test isequal(x0,dtrn.data[1])
@test isequal(y0,dtrn.data[2])

### SAMPLE RUN:

# INFO: Loading MNIST...
#   6.342732 seconds (366.14 k allocations: 502.132 MB, 1.66% gc time)
# INFO: Testing simple mlp
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.01) = (0.37387532f0,18.511799f0,2.8433793f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.21288027f0,0.9327666666666666)
# (test(net,dtst),accuracy(net,dtst)) = (0.2148458f0,0.9289)
# (1,0.9289,0.9327666666666666)
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.01) = (0.14995994f0,22.26936f0,3.9932733f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.13567321f0,0.9574)
# (test(net,dtst),accuracy(net,dtst)) = (0.14322147f0,0.9546)
# (2,0.9546,0.9574)
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.01) = (0.10628127f0,24.865437f0,3.5134742f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.100041345f0,0.9681833333333333)
# (test(net,dtst),accuracy(net,dtst)) = (0.114785746f0,0.9641)
# (3,0.9641,0.9681833333333333)
#  10.178136 seconds (11.31 M allocations: 516.861 MB, 1.27% gc time)
