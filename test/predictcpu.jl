using HDF5
using KUnet
blas_set_num_threads(20)
@time x = h5read(ARGS[1], "/data")
@time l1 = KUnet.Layer(ARGS[2], gpu=false)
@time l2 = KUnet.Layer(ARGS[3], gpu=false)
net = [l1,l2]
@time gc()
@time y = KUnet.predict(net, x, batch=937)
@time y = KUnet.predict(net, x, batch=937)
# Profile.print()
@time h5write(ARGS[4], "data", y)
