using HDF5,JLD,KUnet
KUnet.gpu(false)
isdefined(:xtrn) || (@date @load "zn11oparse1.jld")

d0 = 6f0
c0 = 1f0
g0 = 10f0
nc = size(ytrn,1)
niters=50
nbatch=128
net1=net2=net3=nothing

for i=1:2
@show net1 = Layer[KPerceptron(nc, KUnet.klinear0)]
@date train(net1, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 2.249490525 seconds (984203260 bytes allocated, 19.72% gc time)
@show size(net1[1].s)                                     # size(net1[1].s) => (283246,5341)
end

for i=1:2
@show net1 = Layer[KPerceptron(nc, KUnet.klinear)]
@date train(net1, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 2.006203976 seconds (481354216 bytes allocated, 11.61% gc time)
@show size(net1[1].s)                                     # size(net1[1].s) => (283246,5341)
end

for i=1:2
@show net2 = Layer[KPerceptron(nc, KUnet.kpoly0, [c0,d0])] 
@date train(net2, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 3.646227942 seconds (1101788224 bytes allocated, 12.16% gc time)
@show size(net2[1].s)                                     # size(net2[1].s) => (283246,5248)
end

for i=1:2
@show net2 = Layer[KPerceptron(nc, KUnet.kpoly1, [c0,d0])]
@date train(net2, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 3.433380693 seconds (606331904 bytes allocated, 8.84% gc time)
@show size(net2[1].s)                                     # size(net2[1].s) => (283246,5248)
end

for i=1:2
@show net2 = Layer[KPerceptron(nc, KUnet.kpoly, [c0,d0])]
@date train(net2, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 3.336944494 seconds (474105748 bytes allocated, 6.95% gc time)
@show size(net2[1].s)                                     # size(net2[1].s) => (283246,5248)
end

for i=1:2
@show net3 = Layer[KPerceptron(nc, KUnet.kgauss0, [g0])]
@date train(net3, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 3.335641623 seconds (1467981384 bytes allocated, 21.92% gc time)
@show size(net3[1].s)                                     # (283246,5039)
end

for i=1:2
@show net3 = Layer[KPerceptron(nc, KUnet.kgauss, [g0])]
@date train(net3, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 2.678601132 seconds (988978008 bytes allocated, 19.58% gc time)
@show size(net3[1].s)                                     # (283246,5039)
end

