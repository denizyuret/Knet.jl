import Base: convert, reshape, fill!, isempty, full, copy!, similar, stride
import Base: Ac_mul_B, A_mul_Bc, Ac_mul_Bc
import Base: A_mul_Bt,  At_mul_B
import Base: A_mul_Bt!, At_mul_B!, A_mul_B!
import Base.LinAlg.BLAS: gemm!, axpy!, scal!
import CUDArt: malloc, free, pitchedptr, rt, to_host
import Compat: unsafe_convert

convert{T,S}(::Type{AbstractCudaArray{T}}, x::Array{S})=CudaDynArray(convert(Array{T}, x))
convert{T,S}(::Type{Array{T}}, x::AbstractCudaArray{S})=convert(Array{T}, to_host(x))
reshape(a::AbstractCudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
reshape(a::AbstractCudaArray, dims::Int...)=reshape(a, dims)
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
scal!{T}(n::Integer,alpha::T,x::AbstractCudaArray{T},incx::Integer)=(scal!(n,alpha,convert(CudaArray,x),incx);x)

# This is a function blas does not have but frequently needed:
axpb!(n,a,b,x::AbstractCudaArray{Float32})=(ccall((:axpb32,libkunet),Void,(Cint,Cfloat,Cfloat,Ptr{Cfloat}),n,a,b,x); x)
axpb!(n,a,b,x::AbstractCudaArray{Float64})=(ccall((:axpb64,libkunet),Void,(Cint,Cdouble,Cdouble,Ptr{Cdouble}),n,a,b,x); x)
axpb!(n,a,b,x::AbstractArray)=(for i=1:length(x); x[i] = a*x[i]+b; end; x)

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
CudaDynArray(a::Array)=CudaDynArray(CudaArray(a))
to_host(a::CudaDynArray)=to_host(convert(CudaArray,a))

free(a::CudaDynArray)=free(pointer(a))
#CudaArray(a::CudaDynArray)=CudaArray(a.ptr, a.dims, a.dev)
convert(::Type{CudaArray}, a::CudaDynArray)=CudaArray(a.ptr, a.dims, a.dev)
similar(a::CudaDynArray,T,dims::Dims)=CudaDynArray(T,dims)
similar(a::CudaDynArray)=CudaDynArray(eltype(a), size(a))
stride(a::CudaDynArray, dim::Integer) = prod(size(a)[1:dim-1])
pitchedptr{T}(a::CudaDynArray{T,2})=rt.cudaPitchedPtr(pointer(a), size(a,1)*sizeof(T), size(a,1), size(a,2))
unsafe_convert{T}(::Type{Ptr{T}}, g::CudaDynArray) = unsafe_convert(Ptr{T}, pointer(g))

function reinterpret{T}(::Type{T}, a::CudaDynArray, dims::Dims)
    oldbytes = a.cap * sizeof(eltype(a))
    newbytes = prod(dims) * sizeof(T)
    @assert newbytes <= oldbytes
    newcap = div(oldbytes, sizeof(T))
    CudaDynArray{T,length(dims)}(a.ptr, dims, a.dev, newcap)
end

# Adjust cap with optional copy
function cap!(a::CudaDynArray, n::Integer; copy=false)
    n <= a.cap && return a      # We never shrink the array.
    a.cap = int(1.3*n+1)        # 1.3 ensures a3 can be written where a0+a1 used to be
    if copy
        b = CudaArray(a.ptr, a.dims, a.dev) # save old contents for copy
        a.ptr = malloc(eltype(a), a.cap)
        copy!(a, 1, b, 1, min(n, prod(a.dims)))
        free(b.ptr)
    else
        free(a.ptr)
        a.ptr = malloc(eltype(a), a.cap)
    end
    return a
end

# size! sets size with optional copy
size!(a::CudaDynArray, d::Dims; copy=false)=(size(a)==d ? a : (a=cap!(a,prod(d); copy=copy); a.dims=d; a))
size!(a::CudaArray, d::Dims; copy=false)=(size(a)==d ? a : size!(CudaDynArray(a), d; copy=copy))
size!(a::DenseArray, d::Dims; copy=false)=(size(a)==d ? a : (b=similar(a,d); copy && copy!(b,1,a,1,min(length(a),prod(d))); b))

size!(a::CudaDynArray, d::Integer...; o...)=size!(a,d; o...)
size!(a::CudaArray, d::Integer...; o...)=size!(a,d; o...)
size!(a::DenseArray, d::Integer...; o...)=size!(a,d; o...)

function hcat!{T}(a::CudaDynArray{T,2}, b::Union(CudaMatrix{T},Matrix{T},CudaDynArray{T,2}), 
                  vj=(1:size(b,2)), nj=length(vj))
    @assert size(a,1) == size(b,1)
    (nrows,ncols) = size(a)
    newlen = length(a) + nj * nrows
    newlen > a.cap && (a = cap!(a, newlen; copy=true))
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

function uniq!(ss::CudaDynArray, uu::CudaDynArray, vv::CudaDynArray)
    (s,u,v)=map(cpucopy,(ss,uu,vv))
    (s,u,v)=uniq!(s,u,v)
    n = size(s,2)
    ss = size!(ss, (size(s,1),n))
    uu = size!(uu, (size(u,1),n))
    vv = size!(vv, (size(v,1),n))
    copy!(ss, 1, s, 1, size(s,1)*n)
    copy!(uu, 1, u, 1, size(u,1)*n)
    copy!(vv, 1, v, 1, size(v,1)*n)
    return (ss,uu,vv)
end

function uniq!(s::AbstractArray, u::AbstractArray, v::AbstractArray)
    ds = Dict{Any,Int}()        # dictionary of support vectors
    ns = 0                      # number of unique support vectors
    for j=1:size(s,2)
        jj = get!(ds, s[:,j], ns+1)
        if jj <= ns
            @assert ns == length(ds) < j
            u[:,jj] += u[:,j]
            v[:,jj] += v[:,j]
        else
            @assert jj == ns+1 == length(ds) <= j
            ns = ns+1
            if jj != j
                s[:,jj] = s[:,j]
                u[:,jj] = u[:,j]
                v[:,jj] = v[:,j]
            end
        end
    end
    @assert ns == length(ds)
    s = size!(s, (size(s,1),ns))
    u = size!(u, (size(u,1),ns))
    v = size!(v, (size(v,1),ns))
    return (s,u,v)
end

# This one has to be defined like this because of a conflict with the CUDArt version:
#fill!(A::AbstractCudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)
fill!(A::CudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)
fill!(A::CudaDynArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)

