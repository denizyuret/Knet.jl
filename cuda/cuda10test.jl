include("cuda10.jl")
using CUDArt
libknet8handle = Libdl.dlopen(Libdl.find_library(["libknet8"],[Pkg.dir("Knet/cuda")]))

SIZE = 1000000
ITER = 10000
x32 = CudaArray(rand(Float32,SIZE))
y32 = similar(x32)
s32 = rand(Float32)
x64 = CudaArray(rand(Float64,SIZE))
y64 = similar(x64)
s64 = rand(Float64)

function cuda10test(fname, jname=fname, o...)
    println(fname)
    fcpu = eval(parse(jname))
    f32 = Libdl.dlsym(libknet8handle, fname*"_32_10")
    @time cuda10rep(f32,x32,s32,y32)
    isapprox(to_host(y32),fcpu(to_host(x32),s32)) || warn("$fname 32")
    f64 = Libdl.dlsym(libknet8handle, fname*"_64_10")
    @time cuda10rep(f64,x64,s64,y64)
    isapprox(to_host(y64),fcpu(to_host(x64),s64)) || warn("$fname 64")
end

function cuda10rep{T}(f,x::CudaArray{T},s::T,y::CudaArray{T})
    n = Cint(length(y))
    for i=1:ITER
        ccall(f,Void,(Cint,Ptr{T},T,Ptr{T}),n,x,s,y)
    end
    device_synchronize()
    CUDArt.rt.checkerror(CUDArt.rt.cudaGetLastError())
end

for f in cuda10
    isa(f,Tuple) || (f=(f,))
    cuda10test(f...)
end
