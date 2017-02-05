using Knet
libknet8handle = Libdl.dlopen(Knet.libknet8)

SIZE = 100000
ITER = 100000
x32 = KnetArray(rand(Float32,SIZE))
y32 = KnetArray(rand(Float32,SIZE))
z32 = similar(x32)
x64 = KnetArray(rand(Float64,SIZE))
y64 = KnetArray(rand(Float64,SIZE))
z64 = similar(x64)

function cuda11test(fname, jname=fname, o...)
    println(fname)
    fcpu = eval(parse(jname))
    f32 = Libdl.dlsym(libknet8handle, fname*"_32_11")
    @time cuda11rep(f32,x32,y32,z32)
    isapprox(Array(z32),fcpu(Array(x32),Array(y32))) || warn("$fname 32")
    f64 = Libdl.dlsym(libknet8handle, fname*"_64_11")
    @time cuda11rep(f64,x64,y64,z64)
    isapprox(Array(z64),fcpu(Array(x64),Array(y64))) || warn("$fname 64")
end

function cuda11rep{T}(f,x::KnetArray{T},y::KnetArray{T},z::KnetArray{T})
    n = Cint(length(z))
    for i=1:ITER
        ccall(f,Void,(Cint,Ptr{T},Ptr{T},Ptr{T}),n,x,y,z)
    end
    Knet.@cuda(cudart,cudaDeviceSynchronize,())
    Knet.@cuda(cudart,cudaGetLastError,())
end

for f in Knet.cuda11
    isa(f,Tuple) || (f=(f,))
    cuda11test(f...)
end
