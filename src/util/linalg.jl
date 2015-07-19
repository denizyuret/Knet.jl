import Base: A_mul_B!, A_mul_Bt!, At_mul_B!
using Base.LinAlg.BLAS: gemm! # , axpy!, scal!

### CUDAARRAY

A_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','N',one(T),A,B,zero(T),C)
A_mul_Bt!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','T',one(T),A,B,zero(T),C)
At_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('T','N',one(T),A,B,zero(T),C)

### KUDENSE

# The input could be a tensor or a vector.  In which case perform
# internal calculations in 2D.

mat2d(x)=(ndims(x)==2 ? x : reshape(x, size2(x)))
A_mul_B!{S,T}(C::KUdense{S,T}, A::KUparam{S,T}, B::KUdense{S,T})=(A_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
At_mul_B!{S,T}(C::KUdense{S,T}, A::KUparam{S,T}, B::KUdense{S,T})=(At_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
A_mul_Bt!{S,T}(C::BaseArray{T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_Bt!(mat2d(C), mat2d(A.arr), mat2d(B.arr)); C)

### SPARSE{ARRAY}

function A_mul_B!(k::Matrix, x::Sparse{Array}, s::Sparse{Array}) # 1607
    @assert size(k)==(size(x,1), size(s,2))
    # eltype's do not have to match.
    fill!(k, zero(eltype(k)))
    @inbounds for scol=1:size(s,2)
        @inbounds for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
            srow = s.rowval[sp]
            sval = s.nzval[sp]  # 133
            @inbounds for xp=x.colptr[srow]:(x.colptr[srow+1]-1)
                xrow = x.rowval[xp] # 63
                xval = x.nzval[xp]  # 217
                yinc = xval * sval  # 245
                k[xrow,scol] += yinc # 789
            end
        end
    end
    return k
end

function A_mul_B!(k::Matrix, x::Matrix, s::Sparse{Array}) # 1607
    @assert size(k)==(size(x,1), size(s,2))
    # eltype's do not have to match.
    fill!(k, zero(eltype(k)))
    @inbounds for scol=1:size(s,2)
        @inbounds for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
            sval = s.nzval[sp]  # 133
            srow = s.rowval[sp] # xcol
            @inbounds for xrow=1:size(x,1)
                xval = x[xrow,srow]
                yinc = xval * sval  # 245
                k[xrow,scol] += yinc # 789
            end
        end
    end
    return k
end

### SPARSE{CUDAARRAY}

function At_mul_B!(k::CudaArray{Float32,2}, x::Sparse{CudaArray,Float32,Int32}, s::Sparse{CudaArray,Float32,Int32})
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:At_mul_B_32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    return k
end

function At_mul_B!(k::CudaArray{Float64,2}, x::Sparse{CudaArray,Float64,Int32}, s::Sparse{CudaArray,Float64,Int32})
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:At_mul_B_64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    return k
end

function A_mul_B!(k::CudaArray{Float32,2}, x::Sparse{CudaArray,Float32,Int32}, s::Sparse{CudaArray,Float32,Int32})
    @assert size(k)==(size(x,1),size(s,2))
    ccall((:A_mul_B_32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    return k
end

function A_mul_B!(k::CudaArray{Float64,2}, x::Sparse{CudaArray,Float64,Int32}, s::Sparse{CudaArray,Float64,Int32})
    @assert size(k)==(size(x,1),size(s,2))
    ccall((:A_mul_B_64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    return k
end

### KUSPARSE

A_mul_B!{A}(k::KUdense{A}, x::KUsparse{A}, s::KUsparse{A})=
    A_mul_B!(convert(A, k), convert(Sparse, x), convert(Sparse, s))

At_mul_B!{A}(k::KUdense{A}, x::KUsparse{A}, s::KUsparse{A})=
    At_mul_B!(convert(A, k), convert(Sparse, x), convert(Sparse, s))

A_mul_B!{A}(k::KUdense{A}, x::KUdense{A}, s::KUsparse{A})=
    A_mul_B!(convert(A, k), convert(A, x), convert(Sparse, s))

At_mul_B!{A}(k::KUdense{A}, x::KUdense{A}, s::KUsparse{A})=
    At_mul_B!(convert(A, k), convert(A, x), convert(Sparse, s))

