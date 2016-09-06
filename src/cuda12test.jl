using Knet,CUDArt
libknet8handle = Libdl.dlopen(Libdl.find_library(["libknet8"],[Pkg.dir("Knet/src")]))

SIZE1 = 1000
SIZE2 = 100
ITER = 100000
mat32 = KnetArray(rand(Float32,SIZE1,SIZE2))
mat32b = KnetArray(rand(Float32,SIZE1,SIZE2))
col32 = KnetArray(rand(Float32,SIZE1))
row32 = KnetArray(rand(Float32,1,SIZE2))
out32 = similar(mat32)
mat64 = KnetArray(rand(Float64,SIZE1,SIZE2))
mat64b = KnetArray(rand(Float64,SIZE1,SIZE2))
col64 = KnetArray(rand(Float64,SIZE1))
row64 = KnetArray(rand(Float64,1,SIZE2))
out64 = similar(mat64)
f32 = f64 = nothing

function cuda12test(fname, jname=fname, o...)
    println(fname)
    fcpu = eval(parse(jname))
    global f32 = Libdl.dlsym(libknet8handle, fname*"_32_12")
    @time cuda12rep(f32,mat32,mat32b,out32)
    isapprox(to_host(out32),fcpu(to_host(mat32),to_host(mat32b))) || warn("$fname 32 mat mat")
    @time cuda12rep(f32,mat32,col32,out32)
    isapprox(to_host(out32),fcpu(to_host(mat32),to_host(col32))) || warn("$fname 32 mat col")
    @time cuda12rep(f32,mat32,row32,out32)
    isapprox(to_host(out32),fcpu(to_host(mat32),to_host(row32))) || warn("$fname 32 mat row")
    @time cuda12rep(f32,row32,col32,out32)
    isapprox(to_host(out32),fcpu(to_host(row32),to_host(col32))) || warn("$fname 32 row col")
    global f64 = Libdl.dlsym(libknet8handle, fname*"_64_12")
    @time cuda12rep(f64,mat64,mat64b,out64)
    isapprox(to_host(out64),fcpu(to_host(mat64),to_host(mat64b))) || warn("$fname 64 mat mat")
    @time cuda12rep(f64,mat64,col64,out64)
    isapprox(to_host(out64),fcpu(to_host(mat64),to_host(col64))) || warn("$fname 64 mat col")
    @time cuda12rep(f64,mat64,row64,out64)
    isapprox(to_host(out64),fcpu(to_host(mat64),to_host(row64))) || warn("$fname 64 mat row")
    @time cuda12rep(f64,row64,col64,out64)
    isapprox(to_host(out64),fcpu(to_host(row64),to_host(col64))) || warn("$fname 64 row col")
end

function cuda12rep{T}(f,x::KnetArray{T},y::KnetArray{T},z::KnetArray{T})
    n = Cint(length(z))
    if size(x,1)==1; sx=size(z,1); else; sx=1; end
    if size(y,1)==1; sy=size(z,1); else; sy=1; end
    nx = length(x); ny = length(y)
    for i=1:ITER
        ccall(f,Void,(Cint,Ptr{T},Cint,Cint,Ptr{T},Cint,Cint,Ptr{T}),n,x,sx,nx,y,sy,ny,z)
    end
    device_synchronize()
    CUDArt.rt.checkerror(CUDArt.rt.cudaGetLastError())
end

for f in Knet.cuda12
    isa(f,Tuple) || (f=(f,))
    cuda12test(f...)
end

