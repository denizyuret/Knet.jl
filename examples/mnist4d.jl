# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# 4-D convolution test

using Base.Test
using KUnet
using KUnet: params, isapprox2
isdefined(:MNIST) || include("mnist.jl")
setseed(42)
nbatch=100

dtrn = ItemTensor(reshape(MNIST.xtrn,28,28,1,div(length(MNIST.xtrn),28*28)), MNIST.ytrn; batch=nbatch)
dtst = ItemTensor(reshape(MNIST.xtst,28,28,1,div(length(MNIST.xtst),28*28)), MNIST.ytst; batch=nbatch)

x0 = copy(dtrn.data[1])
y0 = copy(dtrn.data[2])

info("Testing lenet")
lenet = Net(Conv(20,5), Bias(), Relu(), Pool(2),
            Conv(50,5), Bias(), Relu(), Pool(2),
            Mmul(500), Bias(), Relu(),
            Mmul(10), Bias(), Soft(), SoftLoss())
setparam!(lenet; lr=0.1)

# single batch for training for quick debug
dtrn1 = ItemTensor(reshape(MNIST.xtrn,28,28,1,div(length(MNIST.xtrn),28*28)), MNIST.ytrn; batch=nbatch,epoch=nbatch)
test(lenet, dtrn1) # to initialize weights
lenet0 = deepcopy(lenet)
lemlp = deepcopy(lenet.op)
@show map(isequal, params(lenet), params(lemlp))
@date @show train(lenet, dtrn1)
@date train(lemlp, csub(dtrn.data[1],1:100), csub(dtrn.data[2],1:100); batch=nbatch)
@show map(isequal, params(lenet), params(lemlp))
@show map(isapprox, params(lenet), params(lemlp))

@time for epoch=1:3
    @date @show train(lenet, dtrn)
    @date train(lemlp, dtrn.data[1], dtrn.data[2]; batch=nbatch)
    # these fail
    # @show map(isequal, params(lenet), params(lemlp))
    # @show map(isapprox, params(lenet), params(lemlp))
    println((epoch, 
             accuracy(lenet,dtrn),
             accuracy(dtrn.data[2], predict(lemlp, dtrn.data[1])),
             accuracy(lenet,dtst),
             accuracy(dtst.data[2], predict(lemlp, dtst.data[1]))))

end

@test isequal(x0,dtrn.data[1])
@test isequal(y0,dtrn.data[2])

# TODO: test with dropout
    # @show (i,1,map(vecnorm,params(lenet)),map(difnorm,params(lenet)))
    # @show (i,1,map(vecnorm,params(lemlp)),map(difnorm,params(lemlp)))
    # @test all(map(isequal, params(lenet), params(lemlp)))
# @show (0,0,map(vecnorm,params(lenet)),map(difnorm,params(lenet)))
# @show (0,0,map(vecnorm,params(lemlp)),map(difnorm,params(lemlp)))
