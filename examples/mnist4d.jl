# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# 4-D convolution test

using Base.Test
using Knet
isdefined(:MNIST) || include("mnist.jl")

function mnist4d(args=ARGS)
    setseed(42)
    nbatch=100

    dtrn = ItemTensor(reshape(MNIST.xtrn,28,28,1,div(length(MNIST.xtrn),28*28)), MNIST.ytrn; batch=nbatch)
    dtst = ItemTensor(reshape(MNIST.xtst,28,28,1,div(length(MNIST.xtst),28*28)), MNIST.ytst; batch=nbatch)

    info("Testing lenet (convolutional net) on MNIST")
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
    lwg = nothing
    for epoch=1:3
        lwg = train(lenet, dtrn)
        @show (epoch, lwg...)
        @show accuracy(lenet,dtrn)
        @show accuracy(lenet,dtst)
    end
    return lwg
end

!isinteractive() && !isdefined(:load_only) && mnist4d(ARGS)


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

### SAMPLE RUN:

# [dy_052@hpc3010 examples]$ julia mnist4d.jl
# INFO: Loading MNIST...
#   5.658037 seconds (362.46 k allocations: 503.185 MB, 1.83% gc time)
# INFO: Testing lenet
# 2015-10-02T14:32:31 @show (epoch,lwg...)
# (epoch,lwg...) = (1,1.8373411f0,15.339206f0,13.506349f0)
#   0.098889 seconds (49.55 k allocations: 2.261 MB, 4.69% gc time)
# 2015-10-02T14:32:31 @show accuracy(lenet,dtrn)
# accuracy(lenet,dtrn) = 0.8724166666666666
#   1.705410 seconds (1.54 M allocations: 81.031 MB, 1.04% gc time)
# 2015-10-02T14:32:33 @show accuracy(lenet,dtst)
# accuracy(lenet,dtst) = 0.8802
#   0.212678 seconds (196.73 k allocations: 10.577 MB, 2.62% gc time)
# 2015-10-02T14:32:37 @show (epoch,lwg...)
# (epoch,lwg...) = (2,0.16057172f0,17.989243f0,12.976568f0)
#   0.000041 seconds (50 allocations: 1.766 KB)
# 2015-10-02T14:32:37 @show accuracy(lenet,dtrn)
# accuracy(lenet,dtrn) = 0.9530333333333333
#   1.256893 seconds (1.16 M allocations: 63.102 MB, 1.32% gc time)
# 2015-10-02T14:32:38 @show accuracy(lenet,dtst)
# accuracy(lenet,dtst) = 0.9543
#   0.212085 seconds (196.76 k allocations: 10.578 MB, 2.61% gc time)
# 2015-10-02T14:32:43 @show (epoch,lwg...)
# (epoch,lwg...) = (3,0.08003744f0,19.31503f0,8.413661f0)
#   0.000042 seconds (50 allocations: 1.766 KB)
# 2015-10-02T14:32:43 @show accuracy(lenet,dtrn)
# accuracy(lenet,dtrn) = 0.9698833333333333
#   1.254705 seconds (1.16 M allocations: 63.105 MB, 1.32% gc time)
# 2015-10-02T14:32:44 @show accuracy(lenet,dtst)
# accuracy(lenet,dtst) = 0.9698
#   0.211996 seconds (196.76 k allocations: 10.578 MB, 2.60% gc time)
