# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# 4-D convolution test

using Base.Test
using Knet
isdefined(:MNIST) || include("mnist.jl")

setseed(42)
nbatch=100

dtrn = ItemTensor(reshape(MNIST.xtrn,28,28,1,div(length(MNIST.xtrn),28*28)), MNIST.ytrn; batch=nbatch)
dtst = ItemTensor(reshape(MNIST.xtst,28,28,1,div(length(MNIST.xtst),28*28)), MNIST.ytst; batch=nbatch)

x0 = copy(dtrn.data[1])
y0 = copy(dtrn.data[2])

info("Testing lenet")
prog = quote
    x  = input()
    w1 = par(5,5,0,20)
    c1 = conv(w1,x)
    b1 = par(0)
    d1 = add(b1,c1)
    r1 = relu(d1)
    p1 = pool(r1; window=2)
    w2 = par(5,5,0,50)
    c2 = conv(w2,p1)
    b2 = par(0)
    d2 = add(b2,c2)
    r2 = relu(d2)
    p2 = pool(r2; window=2)
    w3 = par(500,0)
    a3 = dot(w3,p2)
    b3 = par(0)
    c3 = add(b3,a3)
    d3 = relu(c3)
    w4 = par(10,0)
    a4 = dot(w4,d3)
    b4 = par(0)
    c4 = add(b4,a4)
    p = softmax(c4)
end
lenet = Net(prog)
setopt!(lenet; lr=0.1)

for epoch=1:3
    @date @show (epoch, train(lenet, dtrn)...)
    @date @show accuracy(lenet,dtrn)
    @date @show accuracy(lenet,dtst)
end


### DEAD CODE:

# single batch for training for quick debug
# dtrn1 = ItemTensor(reshape(MNIST.xtrn,28,28,1,div(length(MNIST.xtrn),28*28)), MNIST.ytrn; batch=nbatch,epoch=nbatch)
# @date @show test(lenet, dtrn1) # to initialize weights
# @date @show train(lenet, dtrn1)
# @test isequal(x0,dtrn.data[1])
# @test isequal(y0,dtrn.data[2])

#using Knet: params, isapprox2

# lenet = Net(Conv(20,5), Bias(), Relu(), Pool(2),
#             Conv(50,5), Bias(), Relu(), Pool(2),
#             Mmul(500), Bias(), Relu(),
#             Mmul(10), Bias(), Soft(), SoftLoss())

#lenet0 = deepcopy(lenet)
#lemlp = deepcopy(lenet.op)
#@show map(isequal, params(lenet), params(lemlp))

#@date train(lemlp, csub(dtrn.data[1],1:100), csub(dtrn.data[2],1:100); batch=nbatch)
#@show map(isequal, params(lenet), params(lemlp))
#@show map(isapprox, params(lenet), params(lemlp))

# TODO: test with dropout
    # @show (i,1,map(vecnorm,params(lenet)),map(difnorm,params(lenet)))
    # @show (i,1,map(vecnorm,params(lemlp)),map(difnorm,params(lemlp)))
    # @test all(map(isequal, params(lenet), params(lemlp)))
# @show (0,0,map(vecnorm,params(lenet)),map(difnorm,params(lenet)))
# @show (0,0,map(vecnorm,params(lemlp)),map(difnorm,params(lemlp)))


    # @date train(lemlp, dtrn.data[1], dtrn.data[2]; batch=nbatch)
    # these fail
    # @show map(isequal, params(lenet), params(lemlp))
    # @show map(isapprox, params(lenet), params(lemlp))
             # accuracy(dtrn.data[2], predict(lemlp, dtrn.data[1])),
             # accuracy(dtst.data[2], predict(lemlp, dtst.data[1]))))
