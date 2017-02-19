using BenchmarkTools, Base.Test, Knet
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end
macro disp(x); :(println($(string(x)));display($(esc(x)));println()); end

using Knet: pool1, poolx, poolx1, im2col!, im2col_dims
sync()=Knet.cudaDeviceSynchronize()

conv4gpu(w,x;o...)=(y=conv4(w,x;o...);sync();y)
mode = 0
ax = rand(100,100,100,10)
kx = KnetArray(ax)
aw = rand(3,3,100,100)
kw = KnetArray(aw)
ky = conv4(kw,kx;mode=mode)
ay = conv4(aw,ax;mode=mode)
@test isapprox(ky,ay)
x2 = similar(ax, im2col_dims(aw,ax,ay))
@disp @benchmark im2col!($aw, $ax, $x2, 1, 0, 0, 1, 1, mode)
@disp @benchmark conv4gpu($kw,$kx;mode=mode)
@disp @benchmark conv4($aw,$ax;mode=mode)
error(:ok)

pool2(x;o...)=(y=pool(x;o...);sync();y)
poolx2(x,y,dy;o...)=(dx=poolx(x,y,dy;o...);sync();dx)
o = [(:mode,0)]
ax = rand(100,100,100,10)
kx = KnetArray(ax)
ay = pool(ax;o...)
ky = pool(kx;o...)
@test isapprox(ay,ky)
adx = poolx(ax,ay,ay;o...)
kdx = poolx2(kx,ky,ky;o...)
@test isapprox(adx,kdx)

@disp @benchmark pool($ax;($o)...)
@disp @benchmark pool2($kx;($o)...)
@disp @benchmark poolx($ax,$ay,$ay;($o)...)
@disp @benchmark poolx2($kx,$ky,$ky;($o)...)
