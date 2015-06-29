length(ARGS)==2 || error("Usage: julia zn11pvc.jl datafile gamma")
using HDF5,JLD,KUnet
KUnet.gpu(true)
@date @load ARGS[1]
g0 = eval(parse(ARGS[2]))

net = Layer[KPerceptron(size(ytrn,1), KUnet.kgauss, [g0])]
for epoch=1:20
    @show epoch
    @date train(net, xtrn, ytrn)
    @show size(net[1].s)
    @date KUnet.uniq!(net[1])
    @show size(net[1].s)
    @date zdev = predict(net,xdev)
    @date ztst = predict(net,xtst)
    @show (epoch, size(net[1].s), accuracy(ydev,zdev), accuracy(ytst,ztst))
end

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
