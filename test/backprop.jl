using CUDArt
using HDF5
using KUnet
# blas_set_num_threads(20)
batch = 937
x = h5read(ARGS[1], "/data")
xx = CudaArray(x[:,1:batch])
l1 = KUnet.Layer(ARGS[2])
l2 = KUnet.Layer(ARGS[3])
net = [l1,l2]
y = h5read(ARGS[4], "/data")
yy = CudaArray(y[:,1:batch])
gc()
@time KUnet.backprop(net, xx, yy)
h5write("$(ARGS[5])1.h5", net[1])
h5write("$(ARGS[5])2.h5", net[2])
@time KUnet.backprop(net, xx, yy)
