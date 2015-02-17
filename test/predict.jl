using HDF5
include("../julia/kunet.jl")
include("h5read_layer.jl")
blas_set_num_threads(20)
@time x = h5read(ARGS[1], "/data")
@time l1 = h5read_layer(ARGS[2])
@time l2 = h5read_layer(ARGS[3])
l1.xforw = KUnet.noop
l2.xforw = KUnet.noop
net = [l1,l2]
@time y = KUnet.predict(net, x, 937)
@time h5write(ARGS[4], "data", y)
