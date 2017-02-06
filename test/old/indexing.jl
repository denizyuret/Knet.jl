using Knet, Base.Test, BenchmarkTools

@show a = reshape(KnetArray([1.0:6.0...]), (2,3))
@show a[2]
@show a[2,3]
@show (a[2]=7;a)
@show (a[2,3]=8;a)
@show a[:]
@show (a[:]=9;a)
@show (a[:]=[11.0:16.0...];a)
@show (a[:]=[21:26...];a)
@show (b=KnetArray(rand(6));a[:]=b;a)
@show (b=KnetArray(rand(0:9,6));a[:]=b;a)
@show a[2:4]
@show (a.ptr, a[2:4].ptr)
@show (a[2:4]=10;a)
@show (a[1:2]=a[5:6];a)
@show (a[1:2]=ones(2);a)
@show (a[5:6]=2ones(Int,2);a)
@show a[:,:]
@show a[1,:]
@show a[:,1:2]
@show a[1,1:2]
@show a[1:2,1:2]
@show (a[:,:]=21;a)
@show (a[1,:]=22;a)
@show (a[:,1:2]=23;a)
@show (a[1,1:2]=24;a)
@show (a[1:2,1:2]=25;a)
@show (a[:,:]=31ones(6);a)
@show (a[1,:]=32ones(3);a)
@show (a[:,1:2]=33ones(4);a)
@show (a[1,1:2]=34ones(2);a)
@show (a[1:2,1:2]=35ones(4);a)
@show (a[:,:]=41ones(Int,6);a)
@show (a[1,:]=42ones(Int,3);a)
@show (a[:,1:2]=43ones(Int,4);a)
@show (a[1,1:2]=44ones(Int,2);a)
@show (a[1:2,1:2]=45ones(Int,4);a)


a = rand(3,3)
v = rand(3)
t = v'
ka = KnetArray(a)
kv = KnetArray(v)
kt = KnetArray(t)
@test hcat(a,a) == Array(hcat(ka,ka))
@test hcat(a,v) == Array(hcat(ka,kv))
@test hcat(v,a) == Array(hcat(kv,ka))
@test hcat(v,v) == Array(hcat(kv,kv))
@test vcat(a,a) == Array(vcat(ka,ka))
@test vcat(a,t) == Array(vcat(ka,kt))
@test vcat(t,a) == Array(vcat(kt,ka))
@test vcat(v,v) == Array(vcat(kv,kv))

a = rand(1000,1000)
v = rand(1000)
t = v'
ka = KnetArray(a)
kv = KnetArray(v)
kt = KnetArray(t)
@show @benchmark hcat(a,a)
@show @benchmark hcat(ka,ka)
@show @benchmark KnetArray(hcat(Array(ka),Array(ka)))
@show @benchmark hcat(a,v)
@show @benchmark hcat(ka,kv)
@show @benchmark KnetArray(hcat(Array(ka),Array(kv)))
@show @benchmark hcat(v,a)
@show @benchmark hcat(kv,ka)
@show @benchmark KnetArray(hcat(Array(kv),Array(ka)))
@show @benchmark hcat(v,v)
@show @benchmark hcat(kv,kv)
@show @benchmark KnetArray(hcat(Array(kv),Array(kv)))
@show @benchmark vcat(a,a)
@show @benchmark vcat(ka,ka)
@show @benchmark KnetArray(vcat(Array(ka),Array(ka)))
@show @benchmark vcat(a,t)
@show @benchmark vcat(ka,kt)
@show @benchmark KnetArray(vcat(Array(ka),Array(kt)))
@show @benchmark vcat(t,a)
@show @benchmark vcat(kt,ka)
@show @benchmark KnetArray(vcat(Array(kt),Array(ka)))
@show @benchmark vcat(v,v)
@show @benchmark vcat(kv,kv)
@show @benchmark KnetArray(vcat(Array(kv),Array(kv)))

