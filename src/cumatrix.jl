import Base: convert, reshape, rand!, fill!, isempty, full, copy!, similar, stride
import Base: Ac_mul_B, A_mul_Bc, Ac_mul_Bc
import Base: A_mul_Bt,  At_mul_B
import Base: A_mul_Bt!, At_mul_B!, A_mul_B!
import Base.LinAlg.BLAS: gemm!, axpy!
import CUDArt: malloc, free, pitchedptr, rt
import Compat: unsafe_convert

convert{T,S}(::Type{AbstractCudaArray{T}}, x::Array{S})=CudaArray(convert(Array{T}, x))
convert{T,S}(::Type{Array{T}}, x::AbstractCudaArray{S})=convert(Array{T}, to_host(x))
reshape(a::AbstractCudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
reshape(a::AbstractCudaArray, dims::Int...)=reshape(a, dims)
rand!(A::AbstractCudaArray{Float32})=(ccall((:randfill32,libkunet),Void,(Cint,Ptr{Float32}),length(A),A); A)
rand!(A::AbstractCudaArray{Float64})=(ccall((:randfill64,libkunet),Void,(Cint,Ptr{Float64}),length(A),A); A)
fill!(A::CudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)
isempty(A::AbstractCudaArray)=(length(A)==0)
full(A::AbstractCudaArray)=A            # this is missing
# similar(A::AbstractCudaArray, dims::Int...) = similar(A,dims) # this is buggy, matches similar(A)

# matmul.jl: Linear algebra extended to CudaArrays (this is partial, todo in cublas)
Ac_mul_B{T<:Real}(A::AbstractCudaMatrix{T}, B::AbstractCudaMatrix{T}) = At_mul_B(A, B)
At_mul_B{T}(A::AbstractCudaMatrix{T}, B::AbstractCudaMatrix{T}) = At_mul_B!(similar(B,(size(A,2),size(B,2))),A, B)
A_mul_Bt!{T}(C::AbstractCudaMatrix{T}, A::AbstractCudaMatrix{T}, B::AbstractCudaMatrix{T})=gemm!('N','T',one(T),A,B,zero(T),C)
At_mul_B!{T}(C::AbstractCudaMatrix{T}, A::AbstractCudaMatrix{T}, B::AbstractCudaMatrix{T})=gemm!('T','N',one(T),A,B,zero(T),C)
A_mul_B!{T}(C::AbstractCudaMatrix{T}, A::AbstractCudaMatrix{T}, B::AbstractCudaMatrix{T})=gemm!('N','N',one(T),A,B,zero(T),C)
gemm!{T}(transA::Char,transB::Char,alpha::T,A::AbstractCudaArray{T},B::AbstractCudaArray{T},beta::T,C::AbstractCudaArray{T})=(gemm!(transA,transB,alpha,convert(CudaArray,A),convert(CudaArray,B),beta,convert(CudaArray,C));C)
axpy!{T}(n::Integer,alpha::T,x::AbstractCudaArray{T},incx::Integer,y::AbstractCudaArray{T},incy::Integer)=(axpy!(n,alpha,convert(CudaArray,x),incx,convert(CudaArray,y),incy);y)

# without this patch, deepcopy does not work on structs with CudaArrays
function Base.deepcopy_internal(x::AbstractCudaArray, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    copy(x)
end

# To support efficient hcat! we need dynamic arrays on gpu
# We just add an extra field to CudaArray

type CudaDynArray{T,N} <: AbstractCudaArray{T,N}
    ptr::CudaPtr{T}
    dims::NTuple{N,Int}
    dev::Int
    cap::Int
end

function CudaDynArray(T::Type, dims::Dims, n = prod(dims))
    @assert n >= prod(dims)
    p = malloc(T, n)
    CudaDynArray{T,length(dims)}(p, dims, device(), n)
end

CudaDynArray(T::Type, dims::Integer...) = CudaDynArray(T, dims)
CudaDynArray(a::CudaArray)=CudaDynArray(a.ptr, a.dims, a.dev, prod(a.dims))

free(a::CudaDynArray)=free(pointer(a))
#CudaArray(a::CudaDynArray)=CudaArray(a.ptr, a.dims, a.dev)
convert(::Type{CudaArray}, a::CudaDynArray)=CudaArray(a.ptr, a.dims, a.dev)
similar(a::CudaDynArray,T,dims::Dims)=CudaDynArray(T,dims)
similar(a::CudaDynArray)=CudaDynArray(eltype(a), size(a))
stride(a::CudaDynArray, dim::Integer) = prod(size(a)[1:dim-1])
pitchedptr{T}(a::CudaDynArray{T,2})=rt.cudaPitchedPtr(pointer(a), size(a,1)*sizeof(T), size(a,1), size(a,2))
unsafe_convert{T}(::Type{Ptr{T}}, g::CudaDynArray) = unsafe_convert(Ptr{T}, pointer(g))

# size!: resize if necessary without copy.  
function size!(a::CudaDynArray, n::Integer)
    n <= a.cap && return a      # We never shrink the array.
    a.cap = int(1.3*n+1)        # 1.3 ensures a3 can be written where a0+a1 used to be
    free(a.ptr)
    a.ptr = malloc(eltype(a), a.cap)
    return a
end

size!(a::CudaDynArray, d::Dims)=(size(a)==d ? a : (a=size!(a,prod(d)); a.dims=d; a))

# resize!: resize if necessary with copy
function resize!{T}(a::CudaDynArray{T}, n::Integer)
    n <= a.cap && return a      # We never shrink the array.
    a.cap = int(1.3*n+1)        # 1.3 ensures a3 can be written where a0+a1 used to be
    b = CudaArray(a.ptr, a.dims, a.dev) # save old contents for copy
    a.ptr = malloc(eltype(a), a.cap)
    copy!(a, 1, b, 1, min(n, prod(a.dims)))
    free(b.ptr)                 # free comes after malloc/copy
    return a
end

resize!(a::CudaDynArray, d::Dims)=(size(a)==d ? a : (a=resize!(a,prod(d)); a.dims=d; a))

function hcat!{T}(a::CudaDynArray{T,2}, b::Union(CudaMatrix{T},Matrix{T},CudaDynArray{T,2}), vj::Vector, nj::Integer)
    @assert size(a,1) == size(b,1)
    (nrows,ncols) = size(a)
    newlen = length(a) + nj * nrows
    newlen > a.cap && (a = resize!(a, newlen))
    na = length(a) + 1
    a.dims = (nrows, ncols + nj)
    for i=1:nj
        nb = (vj[i]-1)*nrows+1
        copy!(a, na, b, nb, nrows)
        na += nrows
    end
    gpusync()
    return a
end

