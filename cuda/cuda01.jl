using CUDArt
importall Base

cuda01 = [
("add",".+","s+x[i]"),
("sub",".-","s-x[i]"),
("mul",".*","s*x[i]"),
("div","./","s/x[i]"),
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

# For array,scalar we can get some for free:
# Only type corrected number,array need implementing for basic arithmetic:
(.+){T}(a::CudaArray{T},s::Number)=(.+)(T(s),a)
(.+){T}(s::Number,a::CudaArray{T})=(.+)(T(s),a)
(.-){T}(a::CudaArray{T},s::Number)=(.+)(T(-s),a)
(.-){T}(s::Number,a::CudaArray{T})=(.-)(T(s),a)
(.*){T}(a::CudaArray{T},s::Number)=(.*)(T(s),a)
(.*){T}(s::Number,a::CudaArray{T})=(.*)(T(s),a)
(./){T}(a::CudaArray{T},s::Number)=(.*)(T(1/s),a)
(./){T}(s::Number,a::CudaArray{T})=(./)(T(s),a)

# familiar aliases for broadcasting operations of array & scalar (#7226):
(+){T}(a::CudaArray{T},s::Number)=(.+)(T(s),a)
(+){T}(s::Number,a::CudaArray{T})=(.+)(T(s),a)
(-){T}(a::CudaArray{T},s::Number)=(.+)(T(-s),a)
(-){T}(s::Number,a::CudaArray{T})=(.-)(T(s),a)
(*){T}(a::CudaArray{T},s::Number)=(.*)(T(s),a)
(*){T}(s::Number,a::CudaArray{T})=(.*)(T(s),a)
(/){T}(a::CudaArray{T},s::Number)=(.*)(T(1/s),a)
(\){T}(s::Number,a::CudaArray{T})=(.*)(T(1/s),a)
#(/){T}(s::Number,a::CudaArray{T})=(.*)(T(1/s),a) # not defined in base


function cuda01def(f, j=f, o...)
    libknet8 = Pkg.dir("Knet/cuda/libknet8")
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$(S)_01"
        @eval begin
            function $J(s::$T,x::CudaArray{$T})
                y = similar(x)
                ccall(($F,$libknet8),Void,(Cint,$T,Ptr{$T},Ptr{$T}),length(y),s,x,y)
                return y
            end
        end
    end
end
    
for f in cuda01
    isa(f,Tuple) || (f=(f,))
    cuda01def(f...)
end
