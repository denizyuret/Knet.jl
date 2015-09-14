using CUDArt
using CUDNN
using KUnet
require("mnist.jl")
using MNIST: xtrn, ytrn, xtst, ytst

# load hdf5 file

lr = 0.01

# x1:(28,28,1,64) using 1 channel and 64 minibatch size
@show x1 = Tensor(reshape(MNIST.xtrn[:,1:64], 28, 28, 1, 64))

dims1 = (5,5,1,20)  # using 5x5 filters, 1 input, 20 outputs
conv1 = ConvLayer(w=Filter(float32(randn(dims1)*0.01)),
                  b=Tensor(zeros(Float32,(1,1,dims1[4],1))),
                  pd=PoolingDescriptor((2,2)),
                  pw=UpdateParam(learningRate=lr),
                  pb=UpdateParam(learningRate=2*lr),
                  )

# y1: (24,24,20,64)
# z1: (12,12,20,64) => x2

@show x2=KUnet.forw(conv1,x1)

dims2 = (5,5,20,50)
conv2 = ConvLayer(w=Filter(float32(randn(dims2)*0.01)),
                  b=Tensor(zeros(Float32,(1,1,dims2[4],1))),
                  pd=PoolingDescriptor((2,2)),
                  pw=UpdateParam(learningRate=lr),
                  pb=UpdateParam(learningRate=2*lr),
                  )

# y2: (8,8,50,64)
# z2: (4,4,50,64)
# x3: (800,64)

@show x3=KUnet.forw(conv2,x2)

dims3 = (500,800)
ip3 = Op(w=CudaArray(float32(randn(dims3)*0.01)),
            b=CudaArray(zeros(Float32, dims3[1], 1)),
            f=relu,
            pw=UpdateParam(learningRate=lr),
            pb=UpdateParam(learningRate=2*lr),
            )

# y3: (500,64) => x4

@show x4=KUnet.forw(ip3,x3)

dims4 = (10,500)
ip4 = Op(w=CudaArray(float32(randn(dims4)*0.01)),
            b=CudaArray(zeros(Float32, dims4[1], 1)),
            f=relu,
            pw=UpdateParam(learningRate=lr),
            pb=UpdateParam(learningRate=2*lr),
            )

@show x5=KUnet.forw(ip4,x4)

net4 = [conv1, conv2, ip3, ip4]
