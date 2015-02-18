using CUDArt
using HDF5
include("../julia/kunet.jl")
@time x = h5read(ARGS[1], "/data")
@time l1 = KUnet.Layer(ARGS[2])
@time l2 = KUnet.Layer(ARGS[3])
l1.xforw = KUnet.noop
l2.xforw = KUnet.noop
l1.w = CudaArray(l1.w)
l1.b = CudaArray(l1.b)
l2.w = CudaArray(l2.w)
l2.b = CudaArray(l2.b)
net = [l1,l2]
@time gc()
@time y = KUnet.predict(net, x, 937)
@time y = KUnet.predict(net, x, 937)
# Profile.print()
@time h5write(ARGS[4], "data", y)
