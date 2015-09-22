### These were missing from CUDArt:

using CUDArt

import Base: isequal, convert, reshape, resize!, copy!, isempty, fill!, pointer, issparse, deepcopy_internal
import CUDArt: to_host

to_host(x)=x                    # so we can use it in general
issparse(::CudaArray)=false

isequal(A::CudaArray,B::CudaArray) = (typeof(A)==typeof(B) && size(A)==size(B) && isequal(to_host(A),to_host(B)))

convert{A<:CudaArray,T,N}(::Type{A}, a::Array{T,N})=CudaArray(a)

reshape(a::CudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
reshape(a::CudaArray, dims::Int...)=reshape(a, dims)

function resize!(a::CudaVector, n::Integer)
    if n < length(a)
        a.dims = (n,)
    elseif n > length(a)
        b = CudaArray(eltype(a), convert(Int,n))
        copy!(b, 1, a, 1, min(n, length(a)))
        free(a.ptr)
        a.ptr = b.ptr
        a.dims = b.dims
    end
    return a
end

# Generalizing low level copy using linear indexing to/from gpu
# arrays:

typealias BaseArray{T,N} Union(Array{T,N},SubArray{T,N},CudaArray{T,N})

function copy!{T<:Real}(dst::BaseArray{T}, di::Integer, 
                        src::BaseArray{T}, si::Integer, 
                        n::Integer; stream=null_stream)
    @assert eltype(src) <: eltype(dst) "$(eltype(dst)) != $(eltype(src))"
    @assert isbits(T)
    if si+n-1 > length(src) || di+n-1 > length(dst) || di < 1 || si < 1
        throw(BoundsError())
    end
    esize = sizeof(eltype(src))
    nbytes = n * esize
    dptr = pointer(dst) + (di-1) * esize
    sptr = pointer(src) + (si-1) * esize
    CUDArt.rt.cudaMemcpyAsync(dptr, sptr, nbytes, CUDArt.cudamemcpykind(dst, src), stream)
    return dst
end

isempty(a::CudaArray)=(length(a)==0)

# This one has to be defined like this because of a conflict with the CUDArt version:
#fill!(A::AbstractCudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)
#fill!(A::CudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)

pointer{T}(x::CudaArray{T}, i::Integer) = pointer(x) + (i-1)*sizeof(T)

convert{A<:CudaArray}(::Type{A}, a::Array)=CudaArray(a)
convert{A<:CudaArray}(::Type{A}, a::SparseMatrixCSC)=CudaArray(full(a))
convert{A<:Array}(::Type{A}, a::CudaArray)=to_host(a)

deepcopy_internal(x::CudaArray, s::ObjectIdDict)=(haskey(s,x)||(s[x]=copy(x));s[x])
cpucopy_internal(x::CudaArray, s::ObjectIdDict)=(haskey(s,x)||(s[x]=to_host(x));s[x])
gpucopy_internal(x::CudaArray, s::ObjectIdDict)=deepcopy_internal(x,s)
gpucopy_internal{T<:Number}(x::Array{T}, s::ObjectIdDict)=(haskey(s,x)||(s[x]=CudaArray(x));s[x])

# AbstractArray methods:

Base.getindex{T}(x::CudaArray{T}, i::Integer)=copy!(T[0], 1, x, i, 1)[1]
Base.setindex!{T}(x::CudaArray{T}, v, i::Integer)=copy!(x, i, T[convert(T,v)], 1, 1)
Base.endof(x::CudaArray)=length(x)
