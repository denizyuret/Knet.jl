using BenchmarkTools, Test, Knet
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

#=
# commit 359d3646 2018-09-22, julia 1.0.0 vs commit 4aa5f92f 2018-08-14, julia 0.6.4
# GPU:V100, CPU:Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz

@test(isapprox(pool(ax; o...), pool(kx; o...))) = Test Passed
@benchmarx(pool($(Expr(:$, :kx)); $(Expr(:$, :o))...)) = Trial(143.338 μs) = Trial(154.221 μs)
@benchmark(pool($(Expr(:$, :ax)); $(Expr(:$, :o))...)) = Trial(3.107 ms) = Trial(3.123 ms)
@test(isapprox(poolx(ax, ap, ap; o...), poolx(kx, kp, kp; o...))) = Test Passed
@benchmarx(poolx($(Expr(:$, :kx)), $(Expr(:$, :kp)), $(Expr(:$, :kp)); $(Expr(:$, :o))...)) = Trial(434.826 μs) = Trial(433.798 μs)
@benchmark(poolx($(Expr(:$, :ax)), $(Expr(:$, :ap)), $(Expr(:$, :ap)); $(Expr(:$, :o))...)) = Trial(60.695 ms) = Trial(48.381 ms) -----
@test(isapprox(conv4(aw, ax; o...), conv4(kw, kx; o...))) = Test Passed
@benchmarx(conv4($(Expr(:$, :kw)), $(Expr(:$, :kx)); $(Expr(:$, :o))...)) = Trial(4.055 ms) = Trial(4.054 ms)
@benchmark(conv4($(Expr(:$, :aw)), $(Expr(:$, :ax)); $(Expr(:$, :o))...)) = Trial(223.892 ms) = Trial(302.637 ms) +++++
@test(isapprox(conv4w(aw, ax, ay; o...), conv4w(kw, kx, ky; o...))) = Test Passed
@benchmarx(conv4w($(Expr(:$, :kw)), $(Expr(:$, :kx)), $(Expr(:$, :ky)); $(Expr(:$, :o))...)) = Trial(4.699 ms) = Trial(4.603 ms)
@benchmark(conv4w($(Expr(:$, :aw)), $(Expr(:$, :ax)), $(Expr(:$, :ay)); $(Expr(:$, :o))...)) = Trial(181.501 ms) = Trial(195.529 ms)
@test(isapprox(conv4x(aw, ax, ay; o...), conv4x(kw, kx, ky; o...))) = Test Passed
@benchmarx(conv4x($(Expr(:$, :kw)), $(Expr(:$, :kx)), $(Expr(:$, :ky)); $(Expr(:$, :o))...)) = Trial(4.450 ms) = Trial(4.430 ms)
@benchmark(conv4x($(Expr(:$, :aw)), $(Expr(:$, :ax)), $(Expr(:$, :ay)); $(Expr(:$, :o))...)) = Trial(311.572 ms) = Trial(276.654 ms) ----
@benchmark(im2col!($(Expr(:$, :aw)), $(Expr(:$, :ax)), $(Expr(:$, :x2)), 1, 0, 0, 1, 1, 0)) = Trial(799.827 μs) = Trial(917.461 μs) ++++
@benchmark(col2im!($(Expr(:$, :aw)), $(Expr(:$, :ax)), $(Expr(:$, :x2)), 1, 0, 0, 1, 1, 0)) = Trial(3.571 ms) = Trial(3.554 ms)
=#
