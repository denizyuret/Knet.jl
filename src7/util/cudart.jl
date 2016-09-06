### These were missing from CUDArt:

using CUDArt
using CUDArt: ContiguousArray

import Base: isequal, convert, reshape, resize!, copy!, isempty, fill!, pointer, issparse, deepcopy_internal

issparse(::CudaArray)=false

# For profiling:
copysync!{T}(dst::ContiguousArray{T}, src::ContiguousArray{T}; stream=null_stream)=(copy!(dst,src;stream=stream);gpusync();dst)
fillsync!{T}(X::CudaArray{T}, val; stream=null_stream)=(fill!(X,val;stream=stream);gpusync();X)

isequal(A::CudaArray,B::CudaArray) = (typeof(A)==typeof(B) && size(A)==size(B) && isequal(to_host(A),to_host(B)))

convert{A<:CudaArray,T,N}(::Type{A}, a::Array{T,N})=CudaArray(a)

reshape(a::CudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
reshape(a::CudaArray, dims::Int...)=reshape(a, dims)

function resize!(a::CudaVector, n::Integer)
    if n < length(a)
        a.dims = (n,)
    elseif n > length(a)
        b = CudaArray(eltype(a), convert(Int,n))
        copysync!(b, 1, a, 1, min(n, length(a)))
        free(a.ptr)
        a.ptr = b.ptr
        a.dims = b.dims
    end
    return a
end

# Generalizing low level copy using linear indexing to/from gpu
# arrays:

typealias BaseArray{T,N} Union{Array{T,N},SubArray{T,N},CudaArray{T,N}}

function copysync!{T<:Real}(dst::BaseArray{T}, di::Integer, 
                            src::BaseArray{T}, si::Integer, 
                            n::Integer; stream=null_stream)
    eltype(src) <: eltype(dst) || error("$(eltype(dst)) != $(eltype(src))")
    isbits(T) || error("$T not a bits type")
    if si+n-1 > length(src) || di+n-1 > length(dst) || di < 1 || si < 1
        throw(BoundsError())
    end
    esize = sizeof(eltype(src))
    nbytes = n * esize
    dptr = pointer(dst) + (di-1) * esize
    sptr = pointer(src) + (si-1) * esize
    CUDArt.rt.cudaMemcpyAsync(dptr, sptr, nbytes, CUDArt.cudamemcpykind(dst, src), stream)
    gpusync(); return dst
end

isempty(a::CudaArray)=(length(a)==0)

# This one has to be defined like this because of a conflict with the CUDArt version:
#fillsync!(A::AbstractCudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);gpusync();A)
#fillsync!(A::CudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);gpusync();A)

pointer{T}(x::CudaArray{T}, i::Integer) = pointer(x) + (i-1)*sizeof(T)

convert{A<:CudaArray}(::Type{A}, a::Array)=CudaArray(a)
convert{A<:CudaArray}(::Type{A}, a::SparseMatrixCSC)=CudaArray(full(a))
convert{A<:Array}(::Type{A}, a::CudaArray)=to_host(a)
convert{A<:SparseMatrixCSC}(::Type{A}, a::CudaArray)=sparse(to_host(a))

deepcopy_internal(x::CudaArray, s::ObjectIdDict)=(haskey(s,x)||(s[x]=copy(x));s[x])
cpucopy_internal(x::CudaArray, s::ObjectIdDict)=(haskey(s,x)||(s[x]=to_host(x));s[x])
gpucopy_internal(x::CudaArray, s::ObjectIdDict)=deepcopy_internal(x,s)
gpucopy_internal{T<:Number}(x::Array{T}, s::ObjectIdDict)=(haskey(s,x)||(s[x]=CudaArray(x));s[x])

# AbstractArray methods:

Base.getindex{T}(x::CudaArray{T}, i::Integer)=copysync!(T[0], 1, x, i, 1)[1]
Base.setindex!{T}(x::CudaArray{T}, v, i::Integer)=copysync!(x, i, T[convert(T,v)], 1, 1)
Base.endof(x::CudaArray)=length(x)

# Finding memory usage:

_getbytes(x::DataType,d)=sizeof(Int)
_getbytes(x::NTuple,d)=sum(map(y->_getbytes(y,d), x))
_getbytes(x::AbstractCudaArray,d)=(haskey(d,x) ? 0 : (d[x]=1; length(x) * sizeof(eltype(x))))
_getbytes(x::Symbol,d)=sizeof(Int)

function _getbytes(x::DenseArray,d) 
    haskey(d,x) && return 0; d[x]=1
    total = sizeof(x)
    if !isbits(eltype(x))
        for i = 1:length(x)
            isassigned(x,i) || continue
            isize = _getbytes(x[i],d)
            # @show (typeof(x), i, isize)
            total += isize
        end
    end
    # @show (typeof(x), total)
    # @show (eltype(x), size(x), total)
    return total
end

function _getbytes(x,d)
    total = sizeof(x)
    isbits(x) && return total
    haskey(d,x) && return 0; d[x]=1
    fieldNames = fieldnames(x)
    if fieldNames != ()
        for fieldName in fieldNames
            isdefined(x, fieldName) || continue
            f = x.(fieldName)
            fieldBytes = _getbytes(f,d)
            fieldSize = (isa(f, Union{AbstractArray,CudaArray}) ? (eltype(f),size(f)) : ())
            # @show (typeof(x), fieldName, fieldSize, fieldBytes)
            total += fieldBytes
        end
    end
    return total
end

getbytes(x)=_getbytes(x, ObjectIdDict())

# Changing the way CudaArray prints:

if false # this takes up extra memory
# if !isdefined(:_CudaArray)
    import Base: size, linearindexing, getindex, writemime, summary
    using Base: with_output_limit, showarray, dims2string
    type _CudaArray{T,N} <: AbstractArray{T,N}; a::CudaArray{T,N}; end
    _CudaArray{T,N}(a::CudaArray{T,N})=_CudaArray{T,N}(a)
    size(a::_CudaArray)=size(a.a)
    linearindexing(::_CudaArray)=Base.LinearFast()
    getindex{T}(a::_CudaArray{T},i::Int)=getindex(a.a,i)
    summary{T,N}(a::_CudaArray{T,N})=string(dims2string(size(a)), " CudaArray{$T,$N}")
    writemime(io::IO, ::MIME"text/plain", v::CudaArray)=with_output_limit(()->showarray(io, _CudaArray(v), header=true, repr=false))
end


# To be able to load/save CudaArrays:
if !isdefined(:_CudaArraySave)
    using JLD
    type _CudaArraySave; a::Array; end
    JLD.writeas(c::CudaArray) = _CudaArraySave(to_host(c))
    JLD.readas(d::_CudaArraySave) = CudaArray(d.a)
end
