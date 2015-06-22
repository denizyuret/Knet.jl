using HDF5,JLD,KUnet
isdefined(:xtrn) || (@date @load "zn11oparse1.jld")

d0 = 6f0
c0 = 1f0
g0 = 10f0
nc = size(ytrn,1)
niters=50
nbatch=128

if true
for i=1:2
@show net1 = Layer[KPerceptron(nc, KUnet.klinear0)]
@date train(net1, xtrn, ytrn; iters=niters, batch=nbatch)
@show size(net1[1].s)
end

for i=1:2
@show net1 = Layer[KPerceptron(nc, KUnet.klinear1)]
@date train(net1, xtrn, ytrn; iters=niters, batch=nbatch)
@show size(net1[1].s)
end

for i=1:2
@show net1 = Layer[KPerceptron(nc, KUnet.klinear)]
@date train(net1, xtrn, ytrn; iters=niters, batch=nbatch)
@show size(net1[1].s)
end

for i=1:2
@show net2 = Layer[KPerceptron(nc, KUnet.kpoly0, [c0,d0])]
@date train(net2, xtrn, ytrn; iters=niters, batch=nbatch)
@show size(net2[1].s)
end

for i=1:2
@show net2 = Layer[KPerceptron(nc, KUnet.kpoly1, [c0,d0])]
@date train(net2, xtrn, ytrn; iters=niters, batch=nbatch)
@show size(net2[1].s)
end

for i=1:2
@show net2 = Layer[KPerceptron(nc, KUnet.kpoly, [c0,d0])]
@date train(net2, xtrn, ytrn; iters=niters, batch=nbatch)
@show size(net2[1].s)
end

for i=1:2
@show net3 = Layer[KPerceptron(nc, KUnet.kgauss0, [g0])]
@date train(net3, xtrn, ytrn; iters=niters, batch=nbatch)
@show map(size, (xtrn, net3[1].s))
end

for i=1:2
@show net3 = Layer[KPerceptron(nc, KUnet.kgauss, [g0])]
@date train(net3, xtrn, ytrn; iters=niters, batch=nbatch)
@show map(size, (xtrn, net3[1].s))
end

end #if false