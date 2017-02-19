using BenchmarkTools, Base.Test, Knet
# @show uses print, disp uses display which prints out details of benchmark
macro disp(x); :(println($(string(x)));display($(esc(x)));println()); end
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end
macro benchmarx(x); :(@benchmark ($x;sync())); end
macro gpu(x); if gpu()>=0; esc(x); end; end

using Knet: poolx, im2col!, col2im!, im2col_dims, conv4x, conv4w
@gpu sync()=Knet.cudaDeviceSynchronize()
o = []

ax = rand(100,100,100,10)
aw = rand(3,3,100,100)
@gpu kx = KnetArray(ax)
@gpu kw = KnetArray(aw)

@gpu @show @test isapprox(pool(ax;o...), pool(kx;o...))
@gpu @show @benchmarx pool($kx;$o...) # 0.766ms
@show @benchmark pool($ax;$o...) # 3.407ms, 22.362ms (with and without omp on ai-test)

ap = pool(ax;o...)
@gpu kp = pool(kx;o...)
@gpu @show @test isapprox(poolx(ax,ap,ap;o...), poolx(kx,kp,kp;o...))
@gpu @show @benchmarx poolx($kx,$kp,$kp;$o...) # 2.638ms
@show @benchmark poolx($ax,$ap,$ap;$o...) # 30.780ms, 67.446ms

@gpu @show @test isapprox(conv4(aw,ax;o...), conv4(kw,kx;o...))
@gpu @show @benchmarx conv4($kw,$kx;$o...) # 24.614ms
@show @benchmark conv4($aw,$ax;$o...) # 269.747ms, 346.614ms

ay = conv4(aw,ax;o...)
@gpu ky = conv4(kw,kx;o...)
@gpu @show @test isapprox(conv4w(aw,ax,ay;o...), conv4w(kw,kx,ky;o...))
@gpu @show @benchmarx conv4w($kw,$kx,$ky;$o...) # 28.509ms
@show @benchmark conv4w($aw,$ax,$ay;$o...) # 190.270ms, 287.182ms

@gpu @show @test isapprox(conv4x(aw,ax,ay;o...), conv4x(kw,kx,ky;o...))
@gpu @show @benchmarx conv4x($kw,$kx,$ky;$o...) # 45.799ms
@show @benchmark conv4x($aw,$ax,$ay;$o...) # 221.181ms, 245.694ms

x2 = similar(ax, im2col_dims(aw,ax,ay))
@show @benchmark im2col!($aw, $ax, $x2, 1, 0, 0, 1, 1, 0) # 0.326ms, 10.509ms
@show @benchmark col2im!($aw, $ax, $x2, 1, 0, 0, 1, 1, 0) # 3.367ms, 10.144ms

nothing
