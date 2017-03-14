using BenchmarkTools, Base.Test, Knet

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

@show @benchmark (hcat($km,$km);gs()) # 225
@show @benchmark (hcat($km,$kc);gs()) # 122
@show @benchmark (hcat($kc,$km);gs()) # 126
@show @benchmark (hcat($kc,$kc);gs()) # 22
@show @benchmark (vcat($km,$km);gs()) # 602
@show @benchmark (vcat($km,$kr);gs()) # 316
@show @benchmark (vcat($kr,$km);gs()) # 316
@show @benchmark (vcat($kc,$kc);gs()) # 18

nothing
