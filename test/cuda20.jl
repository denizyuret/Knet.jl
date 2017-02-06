# Time	CUBLS32	CUBLS64	KN32	KN64	CPU32	CPU64	AF32	AF64
# sum	-	-	2.90	3.00	1.91	3.50	3.29	4.41
# prod	-	-	2.89	2.99	25.77	36.54	3.49	4.48
# max	3.77	3.88	2.92	3.01	6.25	8.00	3.53	4.43
# min	3.77	3.72	2.92	3.02	6.24	6.43	3.56	4.41
# asum	3.54	3.62	2.89	2.98	1.18	2.31	-	-
# nrmsq	5.45	9.30	2.88	2.98	3.35	3.13	5.51	7.41

# (*) BLK=128, THR=128 does best for kn.
# (*) what is the secret behind the cpu sum?

using Knet

libknet8handle = Libdl.dlopen(Knet.libknet8)

SIZE = 100000
ITER = 100000
x32 = KnetArray(rand(Float32,SIZE))
x64 = KnetArray(rand(Float64,SIZE))

function cuda20test(fname, jname=fname, o...)
    println(fname)
    fcpu = eval(parse(jname))
    f32 = Libdl.dlsym(libknet8handle, fname*"_32_20")
    @time y32 = cuda20rep(f32,x32)
    # @time z32 = cpu20rep(fcpu,Array(x32))
    z32 = fcpu(Array(x32))
    isapprox(y32,z32) || warn("$fname 32")
    f64 = Libdl.dlsym(libknet8handle, fname*"_64_20")
    @time y64 = cuda20rep(f64,x64)
    # @time z64 = cpu20rep(fcpu,Array(x64))
    z64 = fcpu(Array(x64))
    isapprox(y64,z64) || warn("$fname 64")
end

function cuda20rep{T}(f,x::KnetArray{T})
    n = Cint(length(x))
    y = 0
    for i=1:ITER
        y = ccall(f,T,(Cint,Ptr{T}),n,x)
    end
    Knet.@cuda(cudart,cudaDeviceSynchronize,())
    Knet.@cuda(cudart,cudaGetLastError,())
    return y
end

function cpu20rep{T}(f,x::Array{T})
    y = 0
    for i=1:ITER
        y = f(x)
    end
    return y
end

if isdefined(:AFArray)
function af20rep{T}(f,x::AFArray{T})
    y = 0
    for i=1:ITER
        y = f(x)
    end
    return y
end
end

for f in Knet.cuda20
    isa(f,Tuple) || (f=(f,))
    cuda20test(f...)
    cuda20test(f...)
end
