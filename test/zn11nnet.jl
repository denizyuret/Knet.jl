length(ARGS)>=4 || error("Usage: julia zn11nnet.jl datafile lr drop1 drop2 hidden1 hidden2...")
using HDF5,JLD,KUnet
KUnet.gpu(true)
argn = 0
isdefined(:xtrn) || (@date @load ARGS[argn+=1])

lr = eval(parse(ARGS[argn+=1]))
drop1 = eval(parse(ARGS[argn+=1]))
drop2 = eval(parse(ARGS[argn+=1]))

net = Layer[]
while argn < length(ARGS)
    d = isempty(net) ? drop1 : drop2
    d > 0 && (net = push!(net, Drop(d)))
    h = eval(parse(ARGS[argn+=1]))
    net = append!(net, [Mmul(h), Bias(), Relu()])
end
d = isempty(net) ? drop1 : drop2
d > 0 && (net = push!(net, Drop(d)))
h = size(ytrn, 1)
net = append!(net, [Mmul(h), Bias(), XentLoss()])
setparam!(net, adagrad=1e-8, lr=lr)
@show net

for epoch=1:20
    @date train(net, xtrn, ytrn)
    @date zdev = predict(net,xdev)
    @date ztst = predict(net,xtst)
    @show (epoch, accuracy(ydev,zdev), accuracy(ytst,ztst))
end

# for h=7:16; h0=2^h
#     for g=3:8; g0=2f0^-g
#         @show (h0, g0)
#         net = [Mmul(h0), Bias(), Relu(), Mmul(size(ytrn,1)), Bias(), XentLoss()]
#         setparam!(net, adagrad=1e-8, lr=g0)
#         @date train(net, xtrn, ytrn)
#         @date zdev = predict(net,xdev)
#         @date ztst = predict(net,xtst)
#         @show (h0, g0, accuracy(ydev,zdev), accuracy(ytst,ztst))
#     end
# end

# cnet = cpucopy(net)
# @time train(net, xtrn, ytrn; iters=1000)
# net = gpucopy(cnet)
# gc()
# @time @profile train(net, xtrn, ytrn; iters=1000)


# for epoch=1:10
#     @show epoch
#     @date train(net, xtrn, ytrn; iters=1000)
#     @date zdev = predict(net,xdev)
#     @date ztst = predict(net,xtst)
#     @show (epoch, accuracy(ydev,zdev), accuracy(ytst,ztst))
# end


# Use this to optimize gamma:
# isdefined(:xtrn) || (@date @load "zn11cpv.jld")
# niters=2000
# nbatch=100
# nc = size(ytrn,1)
# for g0=0.10:0.01:0.15
# #for g=2:.5:4; g0 = 2f0^-g
#     println("")
#     @show g0
#     net = Layer[KPerceptron(nc, KUnet.kgauss, [g0])]
#     @date train(net, xtrn, ytrn; iters=niters, batch=nbatch)
#     @show size(net[1].s)
#     @date y = predict(net,xdev)
#     @show accuracy(ydev,y)
# end
