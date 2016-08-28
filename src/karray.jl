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
type KnetArray{T,N} # <: AbstractArray{T,N}
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

# These are defined for AbstractArrays:
Base.length(a::KnetArray)=prod(size(a))
Base.size(a::KnetArray,i::Int)=size(a)[i]

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

# GPU memory allocation is very expensive.  So we create an
# application specific memory manager.  Typically same type, size, and
# number of arrays are needed in every iteration of training and
# testing.  During training these arrays should not be overwritten
# until the backward pass.  During testing they should be recycled as
# much as possible.

# We will have a Dict((arrayType,arrayLength)=>arrays):

!isdefined(:TmpDict) && (TmpDict = Dict())

# Each value in the dict will hold arrays of same type and length with
# idx pointing to the last one used:

type TmpStack; arr::Vector; idx::Int; TmpStack()=new([],0); end

# When we are done with an iteration, we reset idx=0 instead of
# freeing the arrays so we can reuse them:

tmpfree()=(for s in values(TmpDict); s.idx=0; end)

# This is the main function, to be used like "similar":

function tmplike(a, dims::Dims=size(a))
    s = get!(TmpStack, TmpDict, (typeof(a),prod(dims)))
    s.idx += 1
    if s.idx > length(s.arr)
        push!(s.arr, similar(a,dims))
        s.idx > length(s.arr) && error("short stack")
    end
    if size(s.arr[s.idx]) != dims
        s.arr[s.idx] = reshape(s.arr[s.idx], dims)
    end
    return s.arr[s.idx]
end

tmpmem()=[(t...,length(s.arr)) for (t,s) in TmpDict]

function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    nbytes=convert(Int,mfree[1])
    narray=length(CUDArt.cuda_ptrs)
    (nbytes,narray)
end
