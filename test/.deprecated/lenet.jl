require("mnist.jl")
using CUDArt
using KUnet
using MNIST: xtrn, ytrn, xtst, ytst
accuracy(y,z)=mean(findmax(y,1)[2] .== findmax(z,1)[2])
KUnet.srandom(1)

net = [Conv(5,5,1,20), Bias(20), Relu(), Pool(2),
       Conv(5,5,20,50), Bias(50), Relu(), Pool(2),
       Mmul(500,800), Bias(500), Relu(),
       Mmul(10,500), Bias(10), XentLoss()]

@show x = CudaArray(reshape(xtrn[:,1:64], 28, 28, 1, 64))
for l in net
    @show x = KUnet.forw(l, x)
end
@show dy = CudaArray(ytrn[:,1:64])
@show loss(net[length(net)], dy)
for i=length(net):-1:1
    @show dy = KUnet.back(net[i], dy)
end

xtrn = reshape(xtrn, 28, 28, 1, size(xtrn, 2))
xtst = reshape(xtst, 28, 28, 1, size(xtst, 2))
setparam!(net; :lr=0.01)
for i=1:100
    train(net, xtrn, ytrn)
    println((i, accuracy(ytst, predict(net, xtst)), 
                accuracy(ytrn, predict(net, xtrn))))
end

# lr = 0.01

# # x1:(28,28,1,64) using 1 channel and 64 minibatch size
# @show x1 = Tensor(reshape(MNIST.xtrn[:,1:64], 28, 28, 1, 64))

# dims1 = (5,5,1,20)  # using 5x5 filters, 1 input, 20 outputs
# conv1 = ConvLayer(w=Filter(float32(randn(dims1)*0.01)),
#                   b=Tensor(zeros(Float32,(1,1,dims1[4],1))),
#                   pd=PoolingDescriptor((2,2)),
#                   pw=UpdateParam(learningRate=lr),
#                   pb=UpdateParam(learningRate=2*lr),
#                   )

# # y1: (24,24,20,64)
# # z1: (12,12,20,64) => x2

# @show x2=KUnet.forw(conv1,x1)

# dims2 = (5,5,20,50)
# conv2 = ConvLayer(w=Filter(float32(randn(dims2)*0.01)),
#                   b=Tensor(zeros(Float32,(1,1,dims2[4],1))),
#                   pd=PoolingDescriptor((2,2)),
#                   pw=UpdateParam(learningRate=lr),
#                   pb=UpdateParam(learningRate=2*lr),
#                   )

# # y2: (8,8,50,64)
# # z2: (4,4,50,64)
# # x3: (800,64)

# @show x3=KUnet.forw(conv2,x2)

# dims3 = (500,800)
# ip3 = Op(w=CudaArray(float32(randn(dims3)*0.01)),
#             b=CudaArray(zeros(Float32, dims3[1], 1)),
#             f=relu,
#             pw=UpdateParam(learningRate=lr),
#             pb=UpdateParam(learningRate=2*lr),
#             )

# # y3: (500,64) => x4

# @show x4=KUnet.forw(ip3,x3)

# dims4 = (10,500)
# ip4 = Op(w=CudaArray(float32(randn(dims4)*0.01)),
#             b=CudaArray(zeros(Float32, dims4[1], 1)),
#             f=relu,
#             pw=UpdateParam(learningRate=lr),
#             pb=UpdateParam(learningRate=2*lr),
#             )

# @show x5=KUnet.forw(ip4,x4)

# net4 = [conv1, conv2, ip3, ip4]
