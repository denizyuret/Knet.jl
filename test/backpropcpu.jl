using HDF5
include("../julia/kunet.jl")
blas_set_num_threads(20)
batch = 937
x = h5read(ARGS[1], "/data")
l1 = KUnet.Layer(ARGS[2])
l2 = KUnet.Layer(ARGS[3])
y = h5read(ARGS[4], "/data")
l1.xforw = KUnet.noop
l2.xforw = KUnet.noop
net = [l1,l2]
xx = x[:,1:batch]
yy = y[:,1:batch]
gc()
@time KUnet.backprop(net, xx, yy)
h5write("$(ARGS[5])1.h5", net[1])
h5write("$(ARGS[5])2.h5", net[2])
@time KUnet.backprop(net, xx, yy)
