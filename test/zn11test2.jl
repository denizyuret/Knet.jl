using HDF5,JLD,KUnet
KUnet.gpu(true)
isdefined(:xtrn) || (@date @load "zn11oparse.jld")
for y in (:ytrn, :ydev, :ytst)
    (@eval $y=full($y))
end # This speeds up the accuracy fn

net = Layer[KPerceptron(size(ytrn,1), KUnet.kgauss, [0.05f0])]
for epoch=1:100
    @show epoch
    @date train(net, xtrn, ytrn)
    gc()
    @date zdev = predict(net,xdev)
    @date ztst = predict(net,xtst)
    @show (epoch, size(net[1].s), accuracy(ydev,zdev), accuracy(ytst,ztst))
end

# g0 = .1f0
# nc = size(ytrn,1)
# niters=100
# nbatch=100
# net=nothing

# # These results are for zn11oparse (the original combined features)
# # 100x100 best at g0=0.0625 g=4
# # 250x100 best at g0=0.0884 g=3.5
# # 500x100 best at 6477 g0=0.0526 g=4.25
# # 1000x100 best at 11172 g0=0.0526 g=4.25
# for g=3.5:0.25:5
#     println("")
#     @show g0 = 2f0^-g
#     net = Layer[KPerceptron(nc, KUnet.kgauss, [g0])]
#     @date train(net, xtrn, ytrn; iters=niters, batch=nbatch)
#     @show size(net[1].s)
#     @date y = predict(net,xdev)
#     @show accuracy(ydev,y)
# end



# ntest=10000

# d0 = 6f0
# c0 = 1f0

# if false
#     for kernel in (
#                    (:kgauss, [g0]),
#                    # (:klinear, nothing),
#                    # (:kpoly, [c0,d0]),
#                    )
#         for loc in (
#                     :gpu,
#                     # :cpu,
#                     )
#             KUnet.gpu(loc==:gpu)
#             for i=1:2
#                 println("")
#                 @show (loc, kernel)
#                 @show net = Layer[KPerceptron(nc, KUnet.(kernel[1]), kernel[2])]
#                 @date train(net, xtrn, ytrn; iters=niters, batch=nbatch)
#                 @show size(net[1].s)
#                 @date y = predict(net,xtst[:,1:ntest])
#                 @show accuracy(ytst[:,1:ntest],y)
#             end
#         end
#     end


#xtrn1=xtrn[:,1:100000]
#ytrn1=ytrn[:,1:100000]
#net1=net2=net3=nothing

#     KUnet.gpu(true)
#     for i=1:2
#         println("")
#         @show net1 = Layer[KPerceptron(nc, KUnet.klinear)]
#         @date train(net1, xtrn, ytrn; iters=niters, batch=nbatch)
#         @show size(net1[1].s)
#         @date y = predict(net1,xtst[:,1:ntest])
#         @show accuracy(ytst[:,1:ntest],y)
#     end

#     for i=1:2
#         println("")
#         @show net1 = Layer[KPerceptron(nc, KUnet.klinear4)]
#         @date train(net1, xtrn, ytrn; iters=niters, batch=nbatch)
#         @show size(net1[1].s)
#         @date y = predict(net1,xtst[:,1:ntest])
#         @show accuracy(ytst[:,1:ntest],y)
#     end

#     # this shows the difference is due to the algorithm, same is observed on the cpu
#     # we get cpu/gpu divergence between 16 and 17: actually it is a difference of algorithms: limit bug? 1838/1843 nsv @17.
#     # turns out the data was corrupt (unsorted rowval), the algorithms are ok
#     KUnet.gpu(false)
#     for i=1:1
#         @show net3 = Layer[KPerceptron(nc, KUnet.klinear)]
#         @date train(net3, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 1.698705077 seconds (541901976 bytes allocated)
#         @show (size(net3[1].s),accuracy(ytst[:,1:1000],predict(net3,xtst[:,1:1000]))) # size(net3[1].s) => (283246,5341)
#     end

#     KUnet.gpu(false)
#     for i=1:1
#         @show net4 = Layer[KPerceptron(nc, KUnet.klinear3)]
#         @date train(net4, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 5.800463573 seconds (762111972 bytes allocated, 1.68% gc time)
#         @show (size(net4[1].s),accuracy(ytst[:,1:1000],predict(net4,xtst[:,1:1000]))) # size(net4[1].s) => (283246,5341)
#     end

#     for i=1:2
#         @show net1 = Layer[KPerceptron(nc, KUnet.klinear0)]
#         @date train(net1, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 2.249490525 seconds (984203260 bytes allocated, 19.72% gc time)
#         @show size(net1[1].s)                                     # size(net1[1].s) => (283246,5341)
#     end

#     for i=1:2
#         @show net1 = Layer[KPerceptron(nc, KUnet.klinear)]
#         @date train(net1, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 2.006203976 seconds (481354216 bytes allocated, 11.61% gc time)
#         @show size(net1[1].s)                                     # size(net1[1].s) => (283246,5341)
#     end

#     for i=1:2
#         @show net2 = Layer[KPerceptron(nc, KUnet.kpoly0, [c0,d0])] 
#         @date train(net2, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 3.646227942 seconds (1101788224 bytes allocated, 12.16% gc time)
#         @show size(net2[1].s)                                     # size(net2[1].s) => (283246,5248)
#     end

#     for i=1:2
#         @show net2 = Layer[KPerceptron(nc, KUnet.kpoly1, [c0,d0])]
#         @date train(net2, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 3.433380693 seconds (606331904 bytes allocated, 8.84% gc time)
#         @show size(net2[1].s)                                     # size(net2[1].s) => (283246,5248)
#     end

#     for i=1:2
#         @show net2 = Layer[KPerceptron(nc, KUnet.kpoly, [c0,d0])]
#         @date train(net2, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 3.336944494 seconds (474105748 bytes allocated, 6.95% gc time)
#         @show size(net2[1].s)                                     # size(net2[1].s) => (283246,5248)
#     end

#     for i=1:2
#         @show net3 = Layer[KPerceptron(nc, KUnet.kgauss0, [g0])]
#         @date train(net3, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 3.335641623 seconds (1467981384 bytes allocated, 21.92% gc time)
#         @show size(net3[1].s)                                     # (283246,5039)
#     end

#     for i=1:2
#         @show net3 = Layer[KPerceptron(nc, KUnet.kgauss, [g0])]
#         @date train(net3, xtrn, ytrn; iters=niters, batch=nbatch) # elapsed time: 2.678601132 seconds (988978008 bytes allocated, 19.58% gc time)
#         @show size(net3[1].s)                                     # (283246,5039)
#     end

# end # if false
