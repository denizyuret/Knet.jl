using Base.Test
using KUnet
using CUDArt

a = rand(5,8)
b = rand(5,18)

a1 = CPUdense(copy(a))
@test @show cslice!(a1,b,3:5).arr == b[:,3:5]

b2 = copy(b)
a2 = CPUdense(copy(a))
@test @show ccopy!(b2,8,a2) == [b[:,1:7] a b[:,16:18]]

a3 = CPUdense(copy(a))
@test @show ccat!(a2, b, 3:5).arr == [a b[:,3:5]]

a4 = GPUdense(copy(a))
@test @show to_host(cslice!(a4,b,3:5).arr) == b[:,3:5]

b5 = copy(b)
a5 = GPUdense(copy(a))
@test @show ccopy!(b5,8,a5) == [b[:,1:7] a b[:,16:18]]

a6 = GPUdense(copy(a))
@test @show to_host(ccat!(a6, b, 3:5).arr) == [a b[:,3:5]]

# cpu -> cpu
c1 = CPUdense(rand(3,5))
d1 = cpucopy(c1)
@show map(summary, (c1,d1))
@test @show c1.arr == d1.arr
@test @show pointer(c1.arr) == pointer(c1.ptr)
@test @show pointer(d1.arr) == pointer(d1.ptr)
@test @show pointer(d1.arr) != pointer(c1.arr)

# cpu -> gpu
c2 = CPUdense(rand(3,5))
d2 = gpucopy(c2)
@show map(summary, (c2,d2))
@test @show c2.arr == to_host(d2.arr)
@test @show pointer(c2.arr) == pointer(c2.ptr)
@test @show pointer(d2.arr) == pointer(d2.ptr)

# gpu -> cpu
c3 = GPUdense(rand(3,5))
d3 = cpucopy(c3)
@show map(summary, (c3,d3))
@test @show to_host(c3.arr) == d3.arr
@test @show pointer(c3.arr) == pointer(c3.ptr)
@test @show pointer(d3.arr) == pointer(d3.ptr)

# gpu -> gpu
c4 = GPUdense(rand(3,5))
d4 = gpucopy(c4)
@show map(summary, (c4,d4))
@test @show to_host(c4.arr) == to_host(d4.arr)
@test @show pointer(c4.arr) == pointer(c4.ptr)
@test @show pointer(d4.arr) == pointer(d4.ptr)
@test @show pointer(d4.arr) != pointer(c4.arr)

using HDF5,JLD

c5 = CPUdense(rand(3,5))
save("testarray.jld", "c5", c5)
x5 = load("testarray.jld")
d5 = x5["c5"]
@show map(summary, (c5,d5))
@test @show c5.arr == d5.arr
@test @show pointer(c5.arr) == pointer(c5.ptr)
@show pointer(d5.arr) == pointer(d5.ptr)
@test @show pointer(d5.arr) != pointer(c5.arr)

c6 = GPUdense(rand(3,5))
save("testarray.jld", "c6", cpucopy(c6))
x6 = load("testarray.jld")
d6 = x6["c6"]
@show map(summary, (c6,d6))
@test @show to_host(c6.arr) == d6.arr
@test @show pointer(c6.arr) == pointer(c6.ptr)
@show pointer(d6.arr) == pointer(d6.ptr)
