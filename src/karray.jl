# This is mostly copied from CUDArt to create an application specific
# memory manager.

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
        ptr = knetMalloc(nbytes)
        kp = KnetPtr(ptr,nbytes)
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
            knetFree(p)
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

KnetArray{T,N}(::Type{T}, dims::NTuple{N,Int})=KnetArray{T,N}(KnetPtr(sizeof(T)*prod(dims)), dims, gpu())
KnetArray(T::Type, dims::Int...)=KnetArray(T,dims)

typealias KnetMatrix{T} KnetArray{T,2}
typealias KnetVector{T} KnetArray{T,1}

Base.convert{A<:KnetArray,T}(::Type{A}, a::Array{T})=knetcopy!(KnetArray(T,size(a)),1,a,1,length(a))
Base.convert{A<:Array,T}(::Type{A}, a::KnetArray{T})=knetcopy!(Array(T,size(a)),1,a,1,length(a))
Base.similar{T}(a::KnetArray{T})=KnetArray(T,size(a))
Base.similar{T}(a::KnetArray{T},dims::Dims)=KnetArray(T,dims)
Base.similar{T}(a::KnetArray{T},dims::Int...)=KnetArray(T,dims)
Base.unsafe_convert{T}(::Type{Ptr{T}}, a::KnetArray) = Base.unsafe_convert(Ptr{T}, pointer(a))
Base.pointer{T}(a::KnetArray{T})=convert(Ptr{T}, a.ptr.ptr)
Base.pointer{T}(a::KnetArray{T},i)=convert(Ptr{T}, a.ptr.ptr + (i-1)*sizeof(T))
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

function knetMalloc(nbytes::Int)
    if gpu() >= 0
        pp = Ptr{Void}[C_NULL]
        CUDArt.rt.cudaMalloc(pp, nbytes)
        return pp[1]
    else
        convert(Ptr{Void}, pointer(Array(UInt8,nbytes)))
    end
end

function knetFree(p::Ptr{Void})
    if gpu() >= 0
        CUDArt.rt.cudaFree(p)
    end
end

# Generalizing low level copy using linear indexing to/from gpu arrays:
# copy!{T}(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer)

typealias Kcopy{T} Union{Array{T},SubArray{T},KnetArray{T}}

function knetcopy!{T}(dest::Kcopy{T}, doffs::Integer, src::Kcopy{T}, soffs::Integer, n::Integer; stream=nothing)
    n == 0 && return dest
    isbits(T) || error("knetcopy! only works for isbits arrays.")
    if n < 0 || soffs < 1 || doffs < 1 || soffs+n-1 > length(src) || doffs+n-1 > length(dest)
        throw(BoundsError())
    end
    if gpu() >= 0
        stream == nothing && (stream = null_stream)
        CUDArt.rt.cudaMemcpyAsync(pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), cudadir(dest, src), stream)
    else
        Base.unsafe_copy!(pointer(dest,doffs), pointer(src,soffs), n)
    end
    return dest
end

cudadir(dstp::KnetPtr, srcp::Ptr) = CUDArt.rt.cudaMemcpyHostToDevice
cudadir(dstp::Ptr, srcp::KnetPtr) = CUDArt.rt.cudaMemcpyDeviceToHost
cudadir(dstp::KnetPtr, srcp::KnetPtr) = CUDArt.rt.cudaMemcpyDeviceToDevice
cudadir(dstp::Ptr, srcp::Ptr) = CUDArt.rt.cudaMemcpyHostToHost

# TODO: this will be fixed when we inherint from AbstractArray.
Base.display(x::KnetArray)=(print("KnetArray ");display(convert(Array,x)))

meminfo()=[(k,v.used,length(v.free)) for (k,v) in KnetFree]

