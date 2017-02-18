using BenchmarkTools, Base.Test, Knet

using Knet: pool4
sync()=Knet.cudaDeviceSynchronize()
ax = rand(100,100,100,10)
kx = KnetArray(ax)
@time pool(ax;mode=1);
@time pool(ax;mode=1);
@time pool4(ax;mode=1);
@time pool4(ax;mode=1);
@time (pool(kx;mode=1);sync())
@time (pool(kx;mode=1);sync())
@show isapprox(pool(ax;mode=1),pool(kx;mode=1))
@show isapprox(pool4(ax;mode=1),pool(kx;mode=1))

# aw = rand(3, 3, 100, 100)
# @time ay = conv4(aw, ax);
# @time ay = conv4(aw, ax);
