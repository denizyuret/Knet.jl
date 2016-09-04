# This is mostly copied from CUDArt to create an application specific
# memory manager.

using CUDArt

# KnetPtr type holds a gpu allocated pointer.

type KnetPtr
    ptr::Ptr{Void}
    len::Int
end

# We use the KnetPtrs type to keep track of allocated pointers:

type KnetPtrs
    used::Int
    free::Array{Ptr{Void},1}
    KnetPtrs()=new(0,Array(Ptr{Void},0))
end

# When Julia gc reclaims a KnetPtr object, the finalizer does not
# actually release the memory, but inserts it in the KnetFree dict
# keyed by length in bytes so it can be reused.

if !isdefined(:KnetFree)
    KnetFree = Dict{Int,KnetPtrs}()
end

function free(p::KnetPtr)
    ptrs = KnetFree[p.len]
    push!(ptrs.free,p.ptr)
end

function KnetPtr(nbytes::Integer)
    ptrs = get!(KnetPtrs,KnetFree,nbytes)
    if !isempty(ptrs.free)
        kp = KnetPtr(pop!(ptrs.free),nbytes)
    else
        pp = Ptr{Void}[C_NULL]
        CUDArt.rt.cudaMalloc(pp, nbytes)
        kp = KnetPtr(pp[1],nbytes)
        ptrs.used += 1
    end
    finalizer(kp, free)
    return kp
end

# If you really want to clean up memory you need to call knetgc:

function knetgc()
    gc_enable(false)
    for v in values(KnetFree)
        for p in v.free
            CUDArt.rt.cudaFree(p)
        end
        v.used -= length(v.free)
        empty!(v.free)
    end
    gc_enable(true)
end

if !isdefined(:KnetArray)
type KnetArray{T,N} # <: AbstractArray{T,N} # commented this out to find leaks.
    ptr::KnetPtr
    dims::NTuple{N,Int}
    dev::Int
end
end

KnetArray{T,N}(::Type{T}, dims::NTuple{N,Int})=KnetArray{T,N}(KnetPtr(sizeof(T)*prod(dims)), dims, device())
KnetArray(T::Type, dims::Int...)=KnetArray(T,dims)

typealias KnetMatrix{T} KnetArray{T,2}
typealias KnetVector{T} KnetArray{T,1}
typealias KA{T,N} KnetArray{T,N}  # for my sanity

Base.convert{A<:KnetArray,T}(::Type{A}, a::Array{T})=knetcopy!(KnetArray(T,size(a)),1,a,1,length(a))
Base.convert{A<:Array,T}(::Type{A}, a::KnetArray{T})=knetcopy!(Array(T,size(a)),1,a,1,length(a))
Base.similar{T}(a::KnetArray{T})=KnetArray(T,size(a))
Base.similar{T}(a::KnetArray{T},dims::Dims)=KnetArray(T,dims)
Base.similar{T}(a::KnetArray{T},dims::Int...)=KnetArray(T,dims)
Base.unsafe_convert{T}(::Type{Ptr{T}}, a::KnetArray) = Base.unsafe_convert(Ptr{T}, pointer(a))
Base.pointer(a::KnetArray)=a.ptr
CUDArt.to_host(a::KnetArray)=convert(Array,a)
Base.reshape{T}(a::KnetArray{T},dims::Dims)=(prod(dims)==length(a)||throw(DimensionMismatch()); KnetArray(a.ptr,dims,a.dev))

# AbstractArray interface
Base.size(a::KnetArray)=a.dims
Base.linearindexing(::KnetArray)=Base.linearfast()
Base.getindex{T}(a::KnetArray{T}, i::Integer)=knetcopy!(T[0], 1, a, i, 1)[1]
Base.setindex!{T}(a::KnetArray{T}, v, i::Integer)=knetcopy!(a, i, T[convert(T,v)], 1, 1)

# These are defined for AbstractArrays, so we can remove it eventually:
Base.length(a::KnetArray)=prod(size(a))
Base.ndims(a::KnetArray)=length(size(a))
Base.size(x::KnetArray,i::Integer)=(if i>ndims(x); 1; else; size(x)[i]; end)
Base.eltype{T}(x::KnetArray{T})=T
Base.stride(x::KnetArray,i::Integer)=(if i>ndims(x); length(x); else; s=1; for n=1:(i-1); s*=size(x,n); end; s; end)
import AutoGrad: sum_outgrads
sum_outgrads{T}(a::KnetArray{T},b::KnetArray{T})=(a+b)

# Generalizing low level copy using linear indexing to/from gpu arrays:

function knetcopy!{T}(dst::Union{Array{T},SubArray{T},KnetArray{T}}, di::Integer, 
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

# These are needed for knetcopy:
Base.pointer(a::KnetArray)=a.ptr.ptr
CUDArt.cudamemcpykind(dstp::KnetPtr, srcp::Ptr) = CUDArt.rt.cudaMemcpyHostToDevice
CUDArt.cudamemcpykind(dstp::Ptr, srcp::KnetPtr) = CUDArt.rt.cudaMemcpyDeviceToHost
CUDArt.cudamemcpykind(dstp::KnetPtr, srcp::KnetPtr) = CUDArt.rt.cudaMemcpyDeviceToDevice

# TODO: this will be fixed when we inherint from AbstractArray.
Base.display(x::KnetArray)=(print("KnetArray ");display(CUDArt.to_host(x)))

function gpuinfo(msg="")
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    nbytes=convert(Int,mfree[1])
    narray=length(CUDArt.cuda_ptrs)
    print("$msg ")
    println((nbytes,[(k,v.used,length(v.free)) for (k,v) in KnetFree]...,:cuda_ptrs,narray))
end

