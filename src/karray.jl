# This is mostly copied from CUDArt to create an application specific
# memory manager.

using CUDArt

# We need a pointer type with a finalizer.  If KnetArray has a
# finalizer shared pointers created by reshape may get garbage
# collected.

if !isdefined(:KnetPtr)
type KnetPtr{T}; ptr::Ptr{T}; end
end

function KnetPtr(T::Type, n::Integer)
    pp = Ptr{Void}[C_NULL]
    CUDArt.rt.cudaMalloc(pp, n)
    p = KnetPtr(Base.unsafe_convert(Ptr{T},pp[1]))
    finalizer(p, x->CUDArt.rt.cudaFree(x.ptr))
    return p
end

if !isdefined(:KnetArray)
type KnetArray{T,N} # <: AbstractArray{T,N} # commented this out to find leaks.
    ptr::KnetPtr{T}
    dims::NTuple{N,Int}
    dev::Int
end
end

KnetArray(T::Type, dims::Dims)=KnetArray(KnetPtr(T,sizeof(T)*prod(dims)), dims, device())

typealias KnetMatrix{T} KnetArray{T,2}
typealias KnetVector{T} KnetArray{T,1}

Base.convert{A<:KnetArray,T}(::Type{A}, a::Array{T})=copysync!(KnetArray(T,size(a)),1,a,1,length(a))
Base.convert{A<:Array,T}(::Type{A}, a::KnetArray{T})=copysync!(Array(T,size(a)),1,a,1,length(a))
Base.similar{T}(a::KnetArray{T})=KnetArray(T,size(a))
Base.similar{T}(a::KnetArray{T},dims::Dims)=KnetArray(T,dims)
Base.unsafe_convert{T}(::Type{Ptr{T}}, a::KnetArray) = Base.unsafe_convert(Ptr{T}, pointer(a))
Base.unsafe_convert{T}(::Type{Ptr{T}}, p::KnetPtr) = Base.unsafe_convert(Ptr{T}, p.ptr)
Base.pointer(a::KnetArray)=a.ptr
CUDArt.to_host(a::KnetArray)=convert(Array,a)
Base.reshape{T}(a::KnetArray{T},dims::Dims)=(prod(dims)==length(a)||throw(DimensionMismatch()); KnetArray(a.ptr,dims,a.dev))
Base.convert(::Type{UInt}, x::KnetPtr) = convert(UInt,x.ptr)
Base.convert{T}(::Type{KnetPtr{T}}, x::Integer) = KnetPtr(convert(Ptr{T},x))
Base.(:+)(x::KnetPtr,y::Integer) = oftype(x, UInt(UInt(x) + y))

# AbstractArray interface
Base.size(a::KnetArray)=a.dims
Base.linearindexing(::KnetArray)=Base.linearfast()
Base.getindex{T}(a::KnetArray{T}, i::Integer)=copysync!(T[0], 1, a, i, 1)[1]
Base.setindex!{T}(a::KnetArray{T}, v, i::Integer)=copysync!(a, i, T[convert(T,v)], 1, 1)

# These are defined for AbstractArrays, so we can remove it eventually:
Base.length(a::KnetArray)=prod(size(a))
Base.ndims(a::KnetArray)=length(size(a))
Base.size(x::KnetArray,i::Integer)=(if i>ndims(x); 1; else; size(x)[i]; end)
Base.eltype{T}(x::KnetArray{T})=T
Base.stride(x::KnetArray,i::Integer)=(if i>ndims(x); length(x); else; s=1; for n=1:(i-1); s*=size(x,n); end; s; end)

# Generalizing low level copy using linear indexing to/from gpu arrays:

function copysync!{T}(dst::Union{Array{T},SubArray{T},KnetArray{T}}, di::Integer, 
                      src::Union{Array{T},SubArray{T},KnetArray{T}}, si::Integer, 
                      n::Integer; stream=null_stream)
    if si+n-1 > length(src) || di+n-1 > length(dst) || di < 1 || si < 1
        throw(BoundsError())
    end
    esize = sizeof(T)
    nbytes = n * esize
    dptr = pointer(dst) + (di-1) * esize
    sptr = pointer(src) + (si-1) * esize
    CUDArt.rt.cudaMemcpyAsync(dptr, sptr, nbytes, CUDArt.cudamemcpykind(dst, src), stream)
    return dst
end

CUDArt.cudamemcpykind(dstp::KnetPtr, srcp::Ptr) = CUDArt.rt.cudaMemcpyHostToDevice
CUDArt.cudamemcpykind(dstp::Ptr, srcp::KnetPtr) = CUDArt.rt.cudaMemcpyDeviceToHost
CUDArt.cudamemcpykind(dstp::KnetPtr, srcp::KnetPtr) = CUDArt.rt.cudaMemcpyDeviceToDevice

# CUBLAS gemm! expects CudaArrays, here is a workaround with shared pointers:
import CUBLAS: gemm!
Base.convert{T,A<:CudaArray}(::Type{A},a::KnetArray{T})=CudaArray(convert(CudaPtr{T},a.ptr),a.dims,a.dev)
Base.convert{T,A<:CudaPtr}(::Type{A},p::KnetPtr{T})=CudaPtr(p.ptr)
gemm!{T}(transA::Char,transB::Char,alpha::T,A::KnetMatrix{T},B::KnetMatrix{T},beta::T,C::KnetMatrix{T})=
    (gemm!(transA,transB,alpha,convert(CudaArray,A),convert(CudaArray,B),beta,convert(CudaArray,C)); C)

# TODO: fix this:
Base.display(x::KnetArray)=(print("KnetArray ");display(CUDArt.to_host(x)))
