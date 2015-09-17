using Base.Test
using KUnet
using KUnet: params
include("isapprox.jl")          # TODO: isapprox should be part of array.jl
setseed(42)
nbatch=100

(dtrn,dtst) = MNIST(batch=nbatch)

x0 = copy(dtrn.x[1])            # TODO: should rename x a more descriptive data
y0 = copy(dtrn.x[2])

info("Testing simple mlp")
net = Net(Mmul(64), Bias(), Relu(), 
          Mmul(10), Bias(), XentLoss())
setparam!(net, lr=0.5)

test(net, dtst)                 # to init weights
mlp = deepcopy(net.op)

@time for i=1:3
    @show (l,w,g) = train(net, dtrn; gclip=0, gcheck=100, getloss=true, getnorm=true)
    @show (test(net, dtrn), accuracy(net, dtrn))
    @show (test(net, dtst), accuracy(net, dtst))
    train(mlp, dtrn.x[1], dtrn.x[2]; batch=nbatch)
    @test all(map(isequal, params(net), params(mlp)))
    println((i, accuracy(dtst.x[2], predict(mlp, dtst.x[1])),
                accuracy(dtrn.x[2], predict(mlp, dtrn.x[1]))))
end

@test isequal(x0,dtrn.x[1])
@test isequal(y0,dtrn.x[2])

info("Testing lenet")
lenet = Net(Conv(20,5), Bias(), Relu(), Pool(2),
            Conv(50,5), Bias(), Relu(), Pool(2),
            Mmul(500), Bias(), Relu(),
            Mmul(10), Bias(), XentLoss())
setparam!(lenet; lr=0.1)

# single batch for training
(dtrn1,dtst1) = MNIST(batch=nbatch,epoch=nbatch)

test(lenet, dtrn1) # to initialize weights
lenet0 = deepcopy(lenet)
lemlp = deepcopy(lenet.op)

# @show (0,0,map(vecnorm,params(lenet)),map(difnorm,params(lenet)))
# @show (0,0,map(vecnorm,params(lemlp)),map(difnorm,params(lemlp)))
@show map(isequal, params(lenet), params(lemlp))

@time for i=1:1
    println(train(lenet, dtrn1))
    train(lemlp, csub(dtrn.x[1],1:100), csub(dtrn.x[2],1:100); batch=nbatch)

    # @show (i,1,map(vecnorm,params(lenet)),map(difnorm,params(lenet)))
    # @show (i,1,map(vecnorm,params(lemlp)),map(difnorm,params(lemlp)))
    @show map(isequal, params(lenet), params(lemlp))
    @show map(isapprox, params(lenet), params(lemlp))
    # @test all(map(isequal, params(lenet), params(lemlp)))
    println((i, accuracy(dtrn.x[2], predict(lemlp, dtrn.x[1])),
                accuracy(dtrn.x[2], predict(lemlp, dtrn.x[1]))))
end

@test isequal(x0,dtrn1.x[1])
@test isequal(y0,dtrn1.x[2])

# TODO: test with dropout

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
