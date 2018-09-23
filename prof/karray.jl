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

@show @benchmarkx hcat($km,$km) # 225
@show @benchmarkx hcat($km,$kc) # 122
@show @benchmarkx hcat($kc,$km) # 126
@show @benchmarkx hcat($kc,$kc) # 22
@show @benchmarkx vcat($km,$km) # 602
@show @benchmarkx vcat($km,$kr) # 316
@show @benchmarkx vcat($kr,$km) # 316
@show @benchmarkx vcat($kc,$kc) # 18

nothing
