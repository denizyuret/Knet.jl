using CUDArt

### KUdense parametrized by array type, element type, and ndims:

type KUdense{A,T,N}; arr; ptr; end

### CONSTRUCTORS

KUdense(a)=KUdense{atype(a),eltype(a),ndims(a)}(a, reshape(a, length(a)))
KUdense(A::Type, T::Type, d::Dims)=KUdense(A(T,d))
KUdense(A::Type, T::Type, d::Int...)=KUdense(A,T,d)

import Base: similar
similar{A}(a::KUdense{A}, T, d::Dims)=KUdense(A,T,d)

arr(a::Vector,d::Dims)=pointer_to_array(pointer(a), d)
arr(a::CudaVector,d::Dims)=CudaArray(a.ptr, d, a.dev)

### BASIC ARRAY OPS

for fname in (:eltype, :length, :ndims, :size, :strides, :pointer, :isempty)
    @eval (Base.$fname)(a::KUdense)=$fname(a.arr)
end

for fname in (:size, :stride)
    @eval (Base.$fname)(a::KUdense,n)=$fname(a.arr,n)
end

for fname in (:getindex, :setindex!)
    @eval (Base.$fname)(a::KUdense,n...)=$fname(a.arr,n...)
end

atype{A}(::KUdense{A})=A

### GENERALIZED COLUMN OPS

# We want to support arbitrary dimensional arrays.  When data comes in
# N dimensions, we assume it is an array of N-1 dimensional instances
# and the last dimension gives us the instance count.  We will refer
# to the first N-1 dimensions as generalized "columns" of the
# data. Here "column" refers to the last index of an array,
# i.e. column i corresponds to b[:,:,...,i].  

# Resize factor: 1.3 ensures a3 can be written where a0+a1 used to be
const RF=1.3

# CSLICE!  Returns a slice of array b, with columns specified in range
# r, using the storage in KUarray a.  The element types need to match,
# but the size of a does not need to match, it is adjusted as
# necessary.

function cslice!(a::KUdense, b, r::UnitRange)
    @assert eltype(a)==eltype(b)
    n  = clength(b) * length(r)
    length(a.ptr) >= n || resize!(a.ptr, int(RF*n+1))
    b1 = 1 + clength(b) * (first(r) - 1)
    copy!(a.ptr, 1, b, b1, n)
    a.arr = arr(a.ptr, csize(b, length(r)))
    gpusync()
    return a
end

# CCOPY! Copy n columns from src starting at column si, into dst
# starting at column di.

function ccopy!(dst, di, src::KUdense, si=1, n=ccount(src))
    @assert eltype(dst)==eltype(src)
    @assert csize(dst)==csize(src)
    clen = clength(src)
    d1 = 1 + clen * (di - 1)
    s1 = 1 + clen * (si - 1)
    copy!(dst, d1, src.ptr, s1, clen * n)
    gpusync()
    return dst
end

# CCAT! generalizes append! to multi-dimensional arrays.  Adds the
# ability to specify particular columns to append.

function ccat!(a::KUdense, b, cols=(1:ccount(b)), ncols=length(cols))
    @assert eltype(a)==eltype(b)
    @assert csize(a)==csize(b)
    alen = length(a)
    clen = clength(a)
    n = alen + ncols * clen
    length(a.ptr) >= n || resize!(a.ptr, int(RF*n+1))
    for i=1:ncols
        bidx = (cols[i]-1)*clen + 1
        copy!(a.ptr, alen+1, b, bidx, clen)
        alen += clen
    end
    a.arr = arr(a.ptr, csize(a, ccount(a) + ncols))
    gpusync()
    return a
end


import Base: copy!, copy, similar
function copy!(a::KUdense, b::KUdense)
    @assert issimilar(a,b)
    copy!(a.arr, 1, b.arr, 1, length(b.arr))
    return a
end

function copy!(a::KUdense, b::Union(Array,CudaArray))
    @assert eltype(a)==eltype(b)
    @assert size(a)==size(b)
    copy!(a.arr, 1, b, 1, length(b))
    return a
end

copy(a::KUdense)=copy!(similar(a), a)
similar(a::KUdense)=KUdense(atype(a), eltype(a), size(a))

import Base: resize!
function resize!(a::KUdense, d::Dims)
    n = prod(d)
    n > length(a.ptr) && resize!(a.ptr, int(RF*n+1))
    a.arr = arr(a.ptr, d)
    return a
end

# Need to fix deepcopy so it does not create two arrays for arr and ptr:

cpucopy_internal(x::KUdense{Array},d::ObjectIdDict)=(haskey(d,x) ? d[x] : KUdense(copy(x.arr)))
cpucopy_internal(x::KUdense{CudaArray},d::ObjectIdDict)=(haskey(d,x) ? d[x] : KUdense(to_host(x.arr)))
gpucopy_internal(x::KUdense{Array},d::ObjectIdDict)=(haskey(d,x) ? d[x] : KUdense(CudaArray(x.arr)))
gpucopy_internal(x::KUdense{CudaArray},d::ObjectIdDict)=(haskey(d,x) ? d[x] : KUdense(copy(x.arr)))

import Base: rand!, randn!
randn!(a::KUdense, std, mean)=(randn!(a.arr, std, mean); a)
rand!(a::KUdense)=(rand!(a.arr); a)
# rand!(a::KUdense, x0, x1)=(rand!(a.arr); axpb!(length(a), (x1-x0), x0, a.arr); a)



#import Base: show, display
#show(io::IO,a::KUdense)=show(io,a.arr)
#display(a::KUdense)=display(a.arr)

# import Base: convert
# convert(::Type{CPUdense}, a::Array)=CPUdense(a,reshape(a,length(a)))
# convert(::Type{Array}, a::CPUdense)=a.arr

# should we use bytes for a.ptr?
# we do not have copy between different typed arrays?
# actually cudart.rt.memcpy does that...
# but pointer_to_array would not work.
# so eltype is fixed

# function b2y_old(y, b, r, x)
#     # The output is always dense
#     n = size(x, ndims(x))
#     ys = tuple(size(b)[1:end-1]..., n)
#     (y == nothing) && (y = (isa(x, AbstractSparseArray) ? Array(eltype(x), ys) : similar(x, ys)))
#     @assert size(y) == ys
#     yi = 1 + (first(r) - 1) * stride(y, ndims(y))
#     copy!(y, yi, b, 1, length(b))
#     return y
# end

# function b2y(y, b, r, x)
#     (y == nothing) && (y = Array(eltype(x), csize(b, ccount(x))))
#     @assert ccount(y) == ccount(x)
#     copy!(y, r, b)
#     return y
# end

# # These should eventually disappear:

# function x2b(b, x, r)
#     bs = tuple(size(x)[1:end-1]..., length(r))
#     (b == nothing) && (b = (gpu()?GPUdense:CPUdense)(eltype(x), csize(x, length(r))))
#     ccopy!(b, x, r)
#     return b
# end

# function x2b(b, x::SparseMatrixCSC, r)
#     # TODO: in-place operation
#     # Figure out if b has enough storage
#     # Create a new b if not
#     # Copy columns to from x to b
#     # Copy to gpu if necessary
#     b = x[:,r]
#     gpu() && (b = gpucopy(b); gpusync())
#     return b
# end

# function similar!(l, n, a, T=eltype(a), dims=size(a); fill=nothing)
#     if !isdefined(l,n) || (l.(n) == nothing) || (eltype(l.(n)) != T)
#         if isa(a, AbstractSparseArray)
#             l.(n) = spzeros(T, itype(a), dims...)
#             fill != nothing && fill != 0 && error("Cannot fill sparse with $fill")
#         elseif isa(a, DataType)
#             l.(n) = a(T, dims)
#             fill != nothing && fill!(l.(n), fill)
#         else
#             l.(n) = similar(a, T, dims)
#             fill != nothing && fill!(l.(n), fill)
#         end
#         # @show (:alloc, n, size(l.(n)), length(l.(n)))
#     elseif (size(l.(n)) != dims)
#         # p1 = pointer(l.(n))
#         l.(n) = size!(l.(n), dims)
#         # op = (p1==pointer(l.(n)) ? :size! : :realloc)
#         # op==:realloc && (@show (op, n, size(l.(n)), length(l.(n))))
#     end
#     return l.(n)
# end


