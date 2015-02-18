using HDF5
include("../julia/kunet.jl")
blas_set_num_threads(20)
batch = 937
@time x = h5read(ARGS[1], "/data")
@time l1 = KUnet.Layer(ARGS[2])
@time l2 = KUnet.Layer(ARGS[3])
@time y = h5read(ARGS[4], "/data")
l1.xforw = KUnet.noop
l2.xforw = KUnet.noop
net = [l1,l2]
xx = x[:,1:batch]
yy = y[:,1:batch]
@time gc()
@time KUnet.backprop(net, xx, yy)
@time h5write("$(ARGS[5])1.h5", net[1])
@time h5write("$(ARGS[5])2.h5", net[2])
@time KUnet.backprop(net, xx, yy)
