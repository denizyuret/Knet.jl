using HDF5,JLD,KUnet
isdefined(:xtrn) || (@date @load "zn11v.jld")

g0 = .1f0
nc = size(ytrn,1)
niters=100
nbatch=100
ntest=10000
net=nothing
KUnet.gpu(true)

#for g0=0.15:0.01:0.20
for g=0:5
    println("")
    @show g0 # = 2f0^-g
    net = Layer[KPerceptron(nc, KUnet.kgauss, [g0])]
    @date train(net, xtrn, ytrn; iters=niters, batch=nbatch)
    @show size(net[1].s)
    @date y = predict(net,xdev)
    @show accuracy(ydev,y)
end

# net = Layer[KPerceptron(nc, KUnet.kgauss, [0.12f0])]
# for epoch=1:100
#     @show epoch
#     @date train(net, xtrn, ytrn)
#     gc()
#     @date zdev = predict(net,xdev)
#     @date ztst = predict(net,xtst)
#     @show (epoch, size(net[1].s), accuracy(ydev,zdev), accuracy(ytst,ztst))
# end

