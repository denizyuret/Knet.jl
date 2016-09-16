using Knet
libknet8handle = Libdl.dlopen(Knet.libknet8)

SIZE1 = 1000
SIZE2 = 100
ITER = 100000
mat32b = rand(Float32,SIZE1,SIZE2)
mat32 = KnetArray(mat32b)
col32b = rand(Float32,SIZE1)
col32 = KnetArray(col32b)
row32b = rand(Float32,1,SIZE2)
row32 = KnetArray(row32b)
mat64b = rand(Float64,SIZE1,SIZE2)
mat64 = KnetArray(mat64b)
col64b = rand(Float64,SIZE1)
col64 = KnetArray(col64b)
row64b = rand(Float64,1,SIZE2)
row64 = KnetArray(row64b)

f32 = f64 = nothing

function cuda21test(fname, jname=fname, o...)
    println(fname)
    fcpu = eval(parse(jname))

    global f32 = Libdl.dlsym(libknet8handle, fname*"_32_21")
    print("gpu col32 ")
    @time cuda21rep(f32,mat32,col32)
    print("cpu col32 ")
    @time out32 = fcpu21rep(fcpu,mat32b,col32b)
    isapprox(out32,Array(col32)) || warn("$fname 32 mat col")
    print("gpu row32 ")
    @time cuda21rep(f32,mat32,row32)
    print("cpu row32 ")
    @time out32 = fcpu21rep(fcpu,mat32b,row32b)
    isapprox(out32,Array(row32)) || warn("$fname 32 mat row")

    global f64 = Libdl.dlsym(libknet8handle, fname*"_64_21")
    print("gpu col64 ")
    @time cuda21rep(f64,mat64,col64)
    print("cpu col64 ")
    @time out64 = fcpu21rep(fcpu,mat64b,col64b)
    isapprox(out64,Array(col64)) || warn("$fname 64 mat col")
    print("gpu row64 ")
    @time cuda21rep(f64,mat64,row64)
    print("cpu row64 ")
    @time out64 = fcpu21rep(fcpu,mat64b,row64b)
    isapprox(out64,Array(row64)) || warn("$fname 64 mat row")
end

function cuda21rep{T}(f,x::KnetArray{T},y::KnetArray{T})
    nx = length(x); ny = length(y)
    if size(y,1)==1; sy=size(x,1); else; sy=1; end
    for i=1:ITER
        ccall(f,Void,(Cint,Ptr{T},Cint,Cint,Ptr{T}),nx,x,sy,ny,y)
    end
    Knet.@cuda(cudart,cudaDeviceSynchronize,())
    Knet.@cuda(cudart,cudaGetLastError,())
end

function fcpu21rep{T}(f,x::Array{T},y::Array{T})
    if size(y,1)==1; region=1; else; region=2; end
    out = nothing
    for i=1:1 # ITER
        out = f(x,region)
    end
    return out
end

for f in Knet.cuda21
    isa(f,Tuple) || (f=(f,))
    cuda21test(f...)
end

