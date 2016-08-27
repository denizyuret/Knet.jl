using CUDArt
importall Base

cuda11 = [
("add",".+","x[i]+y[i]"),
("sub",".-","x[i]-y[i]"),
("mul",".*","x[i]*y[i]"),
("div","./","x[i]/y[i]"),
# "hypot",
# "rhypot",
# "atan2",
# "pow",
# "frexp",
# "ldexp",
# "scalbn",
# "scalbln",
# "jn",
# "yn",
# "fmod",
# "remainder",
# "mod",
# "fdim",
]

# familiar aliases for broadcasting operations of array & scalar (#7226):
(+){T}(x::CudaArray{T},y::CudaArray{T})=(.+)(x,y)
(-){T}(x::CudaArray{T},y::CudaArray{T})=(.-)(x,y)
#(*){T}(x::CudaArray{T},y::CudaArray{T})=(.*)(x,y) # This is matmul
#(/){T}(x::CudaArray{T},y::CudaArray{T})=(./)(x,y) # This is another linalg op


function cuda11def(f, j=f, o...)
    libknet8 = Pkg.dir("Knet/cuda/libknet8")
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$(S)_11"
        @eval begin
            function $J(x::CudaArray{$T},y::CudaArray{$T})
                if size(x)==size(y)
                    z = similar(x)
                    ccall(($F,$libknet8),Void,(Cint,$Ptr{T},Ptr{$T},Ptr{$T}),length(z),x,y,z)
                    return z
                else
                    error("Not implemented yet.")
                end
            end
        end
    end
end
    
for f in cuda11
    isa(f,Tuple) || (f=(f,))
    cuda11def(f...)
end
