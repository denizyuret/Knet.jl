### These were missing from CUDArt:

using CUDArt

import Base: reshape
reshape(a::CudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
reshape(a::CudaArray, dims::Int...)=reshape(a, dims)

import Base: resize!
function resize!(a::CudaVector, n::Integer)
    if n < length(a)
        a.dims = (n,)
    elseif n > length(a)
        b = CudaArray(eltype(a), n)
        copy!(b, 1, a, 1, min(n, length(a)))
        free(a.ptr)
        a.ptr = b.ptr
        a.dims = b.dims
    end
    return a
end

# Generalizing low level copy using linear indexing to/from gpu
# arrays:

import Base: copy!
function copy!(dst::Union(Array,CudaArray), di::Integer, 
               src::Union(Array,CudaArray), si::Integer, 
               n::Integer; stream=null_stream)
    @assert eltype(src) <: eltype(dst) "$(eltype(dst)) != $(eltype(src))"
    if si+n-1 > length(src) || di+n-1 > length(dst) || di < 1 || si < 1
        throw(BoundsError())
    end
    esize = sizeof(eltype(src))
    nbytes = n * esize
    dptr = pointer(dst) + (di-1) * esize
    sptr = pointer(src) + (si-1) * esize
    CUDArt.rt.cudaMemcpyAsync(dptr, sptr, nbytes, CUDArt.cudamemcpykind(dst, src), stream)
    gpusync()
    return dst
end

import Base: isempty
isempty(a::CudaArray)=(length(a)==0)

# This one has to be defined like this because of a conflict with the CUDArt version:
#fill!(A::AbstractCudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)
import Base: fill!
fill!(A::CudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)

atype(::CudaArray)=CudaArray

