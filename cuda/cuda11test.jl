include("cuda11.jl")
using CUDArt
libknet8handle = Libdl.dlopen(Libdl.find_library(["libknet8"],[Pkg.dir("Knet/cuda")]))

SIZE = 1000000
ITER = 10000
x32 = CudaArray(rand(Float32,SIZE))
y32 = CudaArray(rand(Float32,SIZE))
z32 = similar(x32)
x64 = CudaArray(rand(Float64,SIZE))
y64 = CudaArray(rand(Float64,SIZE))
z64 = similar(x64)

function cuda11test(fname, jname=fname, o...)
    println(fname)
    fcpu = eval(parse(jname))
    f32 = Libdl.dlsym(libknet8handle, fname*"_32_11")
    @time cuda11rep(f32,x32,y32,z32)
    isapprox(to_host(z32),fcpu(to_host(x32),to_host(y32))) || warn("$fname 32")
    f64 = Libdl.dlsym(libknet8handle, fname*"_64_11")
    @time cuda11rep(f64,x64,y64,z64)
    isapprox(to_host(z64),fcpu(to_host(x64),to_host(y64))) || warn("$fname 64")
end

function cuda11rep{T}(f,x::CudaArray{T},y::CudaArray{T},z::CudaArray{T})
    n = Cint(length(z))
    for i=1:ITER
        ccall(f,Void,(Cint,Ptr{T},Ptr{T},Ptr{T}),n,x,y,z)
    end
    device_synchronize()
    CUDArt.rt.checkerror(CUDArt.rt.cudaGetLastError())
end

for f in cuda11
    isa(f,Tuple) || (f=(f,))
    cuda11test(f...)
end
