using HDF5
using KUnet
@time x = h5read(ARGS[1], "/data")
@time l1 = KUnet.Layer(ARGS[2])
@time l2 = KUnet.Layer(ARGS[3])
net = [l1,l2]
@time gc()
@time y = KUnet.predict(net, x, batch=937)
@time y = KUnet.predict(net, x, batch=937)
# Profile.print()
@time h5write(ARGS[4], "data", y)
