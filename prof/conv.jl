using BenchmarkTools, Base.Test, Knet
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end

using Knet: pool1, poolx, poolx1
sync()=Knet.cudaDeviceSynchronize()
pool2(x;o...)=(y=pool(x;o...);sync();y)
poolx2(x,y,dy;o...)=(dx=poolx(x,y,dy;o...);sync();dx)
o = [(:mode,0)]
ax = bx = rand(100,100,100,10)
kx = KnetArray(ax)
ay = pool(ax;o...)
ky = pool(kx;o...)
@test isapprox(ay,ky)
adx = poolx(ax,ay,ay;o...)
kdx = poolx2(kx,ky,ky;o...)
@test isapprox(adx,kdx)

println("\ncpupool");  display(@benchmark pool($ax;($o)...))
println("\ngpupool");  display(@benchmark pool2($kx;($o)...))
println("\ncpupoolx"); display(@benchmark poolx($ax,$ay,$ay;($o)...))
println("\ngpupoolx"); display(@benchmark poolx2($kx,$ky,$ky;($o)...))

# @date ay = pool(ax;o...)
# @date ay = pool(ax;o...)
# @date by = pool1(bx;o...)
# @date by = pool1(bx;o...)
# @date ky = pool2(kx;o...)
# @date ky = pool2(kx;o...)
# @show isapprox(ay,ky)
# @show isapprox(by,ky)
# @date adx = poolx(ax,ay,ay;o...)
# @date adx = poolx(ax,ay,ay;o...)
# @date bdx = poolx1(bx,by,by;o...)
# @date bdx = poolx1(bx,by,by;o...)
# @date kdx = poolx2(kx,ky,ky;o...)
# @date kdx = poolx2(kx,ky,ky;o...)
# @show isapprox(adx,kdx)
# @show isapprox(bdx,kdx)
