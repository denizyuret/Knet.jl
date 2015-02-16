using HDF5
include("../julia/kunet.jl")
@time x = h5read(ARGS[1], "/data")
# TODO: Replace this with h5read_layer:
@time w1 = h5read(ARGS[2], "/w")
@time b1 = h5read(ARGS[2], "/b")
@time w2 = h5read(ARGS[3], "/w")
@time b2 = h5read(ARGS[3], "/b")
l1 = KUnet.Layer()
l1.w = w1
l1.b = b1
l1.xforw! = KUnet.noop
l1.yforw! = KUnet.reluforw!
l2 = KUnet.Layer()
l2.w = w2
l2.b = b2
l2.xforw! = KUnet.noop
l2.yforw! = KUnet.noop
@time y = KUnet.forw!(l1, x)
@time y = KUnet.forw!(l2, y)
@time h5write(ARGS[4], "data", y)
