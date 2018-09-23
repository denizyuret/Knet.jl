using BenchmarkTools, Test, Knet

m=rand(1000,1000)
c=rand(1000)
r=c'

1                               # μs
@show @benchmark hcat($m,$m)    # 1221
@show @benchmark hcat($m,$c)    # 609
@show @benchmark hcat($c,$m)    # 609
@show @benchmark hcat($c,$c)    # 8
@show @benchmark vcat($m,$m)    # 2317
@show @benchmark vcat($m,$r)    # 671
@show @benchmark vcat($r,$m)    # 672
@show @benchmark vcat($c,$c)    # 1

km = KnetArray(m)
kc = KnetArray(c)
kr = KnetArray(r)
gs() = Knet.cudaDeviceSynchronize()
macro benchmarx(x); :(@benchmark ($x;gs())); end

@show @benchmarx hcat($km,$km) # 225
@show @benchmarx hcat($km,$kc) # 122
@show @benchmarx hcat($kc,$km) # 126
@show @benchmarx hcat($kc,$kc) # 22
@show @benchmarx vcat($km,$km) # 602
@show @benchmarx vcat($km,$kr) # 316
@show @benchmarx vcat($kr,$km) # 316
@show @benchmarx vcat($kc,$kc) # 18

nothing

#=
# commit 359d3646 2018-09-22, julia 1.0.0 vs commit 4aa5f92f 2018-08-14, julia 0.6.4
# GPU:V100, CPU:Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz

@benchmark(hcat($(Expr(:$, :m)), $(Expr(:$, :m)))) = Trial(1.013 ms)	= Trial(938.682 μs)
@benchmark(hcat($(Expr(:$, :m)), $(Expr(:$, :c)))) = Trial(512.029 μs)  = Trial(516.471 μs)
@benchmark(hcat($(Expr(:$, :c)), $(Expr(:$, :m)))) = Trial(493.630 μs)  = Trial(469.592 μs)
@benchmark(hcat($(Expr(:$, :c)), $(Expr(:$, :c)))) = Trial(2.864 μs)    = Trial(7.658 μs)   ++++
@benchmark(vcat($(Expr(:$, :m)), $(Expr(:$, :m)))) = Trial(2.264 ms)    = Trial(2.259 ms)  
@benchmark(vcat($(Expr(:$, :m)), $(Expr(:$, :r)))) = Trial(993.510 μs)  = Trial(761.153 μs) ----
@benchmark(vcat($(Expr(:$, :r)), $(Expr(:$, :m)))) = Trial(990.959 μs)  = Trial(760.438 μs) ----
@benchmark(vcat($(Expr(:$, :c)), $(Expr(:$, :c)))) = Trial(887.162 ns)  = Trial(976.700 ns) ++++
@benchmarx(hcat($(Expr(:$, :km)), $(Expr(:$, :km)))) = Trial(60.120 μs)   = Trial(60.645 μs)
@benchmarx(hcat($(Expr(:$, :km)), $(Expr(:$, :kc)))) = Trial(130.411 μs)  = Trial(197.104 μs) ++++
@benchmarx(hcat($(Expr(:$, :kc)), $(Expr(:$, :km)))) = Trial(135.312 μs)  = Trial(101.766 μs) ----
@benchmarx(hcat($(Expr(:$, :kc)), $(Expr(:$, :kc)))) = Trial(15.883 μs)   = Trial(17.072 μs)
@benchmarx(vcat($(Expr(:$, :km)), $(Expr(:$, :km)))) = Trial(144.315 μs)  = Trial(5.020 s)  ????
@benchmarx(vcat($(Expr(:$, :km)), $(Expr(:$, :kr)))) = Trial(170.106 μs)  = Trial(5.589 s)  ????
@benchmarx(vcat($(Expr(:$, :kr)), $(Expr(:$, :km)))) = Trial(179.613 μs)  = Trial(5.044 s)  ????
@benchmarx(vcat($(Expr(:$, :kc)), $(Expr(:$, :kc)))) = Trial(17.047 μs)   = Trial(17.001 μs)
=#
