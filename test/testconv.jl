using Knet,CUDArt

include("../src/op/conv.jl")
include("../src/op/pool.jl")

x = CudaArray(rand(6,5,4,3))
w = CudaArray(rand(3,3,4,2))
c = conv()
s = Knet.infersize(c, size(w), size(x))
y = CudaArray(Float64, s[end])
Knet.forw(c, w, x, y)
dy = similar(y); rand!(dy)
dw = similar(w)
dx = similar(x)
Knet.back(c, dy, dw, dx; x=(w,x))

p = pool()
s = Knet.infersize(p, size(x))
y = CudaArray(Float64, s[end])
Knet.forw(p, x, y)
dy = similar(y); rand!(dy)
dx = similar(x)
Knet.back(p, dy, dx; x=x, y=y)
