"""

    KnetArray{T}(undef,dims)
    KnetArray(a::AbstractArray)
    Array(k::KnetArray)

Container for GPU arrays that supports most of the AbstractArray
interface.  The constructor allocates a KnetArray in the currently
active device, as specified by `gpu()`.  KnetArrays and Arrays can be
converted to each other as shown above, which involves copying to and
from the GPU memory.  Only Float32/64 KnetArrays are fully supported.

Important differences from the alternative CudaArray are: (1) a custom
memory manager that minimizes the number of calls to the slow
cudaMalloc by reusing already allocated but garbage collected GPU
pointers.  (2) a custom getindex that handles ranges such as `a[5:10]`
as views with shared memory instead of copies.  (3) custom CUDA
kernels that implement elementwise, broadcasting, and reduction
operations.

# Supported functions:

* Indexing: getindex, setindex! with the following index types:
  * 1-D: Real, Colon, OrdinalRange, AbstractArray{Real}, AbstractArray{Bool}, CartesianIndex, AbstractArray{CartesianIndex}, EmptyArray, KnetArray{Int32} (low level), KnetArray{0/1} (using float for BitArray) (1-D includes linear indexing of multidimensional arrays)
  * 2-D: (Colon,Union{Real,Colon,OrdinalRange,AbstractVector{Real},AbstractVector{Bool},KnetVector{Int32}}), (Union{Real,AbstractUnitRange,Colon}...) (in any order)
  * N-D: (Real...)

* Array operations: ==, !=, cat, convert, copy, copyto!, deepcopy,
  display, eachindex, eltype, endof, fill!, first, hcat, isapprox,
  isempty, length, ndims, one, ones, pointer, rand!, randn!, reshape,
  similar, size, stride, strides, summary, vcat, vec, zero.
  (cat(x,y,dims=i) supported for i=1,2.)

* Math operators: (-), abs, abs2, acos, acosh, asin, asinh, atan,
  atanh, cbrt, ceil, cos, cosh, cospi, erf, erfc, erfcinv, erfcx,
  erfinv, exp, exp10, exp2, expm1, floor, log, log10, log1p, log2,
  round, sign, sin, sinh, sinpi, sqrt, tan, tanh, trunc

* Broadcasting operators: (.*), (.+), (.-), (./), (.<), (.<=), (.!=),
  (.==), (.>), (.>=), (.^), max, min.  (Boolean operators generate
  outputs with same type as inputs; no support for KnetArray{Bool}.)

* Reduction operators: countnz, maximum, mean, minimum, prod, sum,
  sumabs, sumabs2, norm.
    
* Linear algebra: (*), axpy!, permutedims (up to 5D), transpose

* Knet extras: relu, sigm, invx, logp, logsumexp, conv4, pool,
  deconv4, unpool, mat, update! (Only 4D/5D, Float32/64 KnetArrays
  support conv4, pool, deconv4, unpool)

# Memory management

Knet models do not overwrite arrays which need to be preserved for
gradient calculation.  This leads to a lot of allocation and regular
GPU memory allocation is prohibitively slow. Fortunately most models
use identically sized arrays over and over again, so we can minimize
the number of actual allocations by reusing preallocated but garbage
collected pointers.

When Julia gc reclaims a KnetArray, a special finalizer keeps its
pointer in a table instead of releasing the memory.  If an array with
the same size in bytes is later requested, the same pointer is reused.
The exact algorithm for allocation is:

1. Try to find a previously allocated and garbage collected pointer in
   the current device. (0.5 μs)

2. If not available, try to allocate a new array using cudaMalloc. (10
   μs)

3. If not successful, try running gc() and see if we get a pointer of
   the right size. (75 ms, but this should be amortized over all
   reusable pointers that become available due to the gc)

4. Finally if all else fails, clean up all saved pointers in the
   current device using cudaFree and try allocation one last
   time. (25-70 ms, however this causes the elimination of all
   reusable pointers)

"""
mutable struct KnetArray{T,N} # <: AbstractArray{T,N} (TODO)
    ptr::KnetPtr
    dims::NTuple{N,Int}
end

# Note: I removed <: AbstractArray{T,N} after I painfully discovered
# some inefficient AbstractArray methods inherited unintentionally.
# It is better to define a few extra methods to keep a tighter control
# on what methods exactly get called for KnetArrays.

# TODO: Let's see if this keeps it under control:
import Base: getindex, setindex!, iterate, IndexStyle
getindex(A::KnetArray,I...)=throw(MethodError(getindex,A,I...))
setindex!(A::KnetArray,I...)=throw(MethodError(setindex!,A,I...))
iterate(A::KnetArray,I...)=throw(MethodError(iterate,A,I...))
IndexStyle(::Type{<:KnetArray})=IndexLinear()
# TODO: do we need more defensive methods here?  broadcasted, materialize etc?

# Aliases:

const KnetMatrix{T} = KnetArray{T,2}
const KnetVector{T} = KnetArray{T,1}
const KnetVecOrMat{T} = Union{KnetVector{T}, KnetMatrix{T}}

# Constructors:
# Internal constructor defines KnetArray{T,N}(ptr,dims)

# These define KnetArray{T,N}(undef,dims) and KnetArray{T,N}(undef,d...)
KnetArray{T,N}(::UndefInitializer, d::Vararg{Int,N}) where {T,N} = KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), d)
KnetArray{T,N}(::UndefInitializer, d::NTuple{N,Int}) where {T,N} = KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), d)
KnetArray{T,N}(::UndefInitializer, d::Vararg{Integer,N}) where {T,N} = KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), convert(NTuple{N,Int},d))
KnetArray{T,N}(::UndefInitializer, d::NTuple{N,Integer}) where {T,N} = KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), convert(NTuple{N,Int},d))

# These define KnetArray{T}(undef,dims) and KnetArray{T}(undef,d...)
KnetArray{T}(::UndefInitializer, d::Vararg{Int,N}) where {T,N} = KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), d)
KnetArray{T}(::UndefInitializer, d::NTuple{N,Int}) where {T,N} = KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), d)
KnetArray{T}(::UndefInitializer, d::Vararg{Integer,N}) where {T,N} = KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), convert(NTuple{N,Int},d))
KnetArray{T}(::UndefInitializer, d::NTuple{N,Integer}) where {T,N} = KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), convert(NTuple{N,Int},d))

# KnetArray(::KnetArray) creates a copy, convert returns an alias if possible
KnetArray(A::KnetArray{T,N})    where {T,N}   = KnetArray{T,N}(A)
KnetArray{T}(A::KnetArray{S,N}) where {T,N,S} = KnetArray{T,N}(A)
KnetArray{T,N}(x::KnetArray{T,N}) where {T,N} = _unsafe_copy!(KnetArray{T}(undef,size(x)), 1, x, 1, length(x))
KnetArray{T,N}(x::KnetArray{S,N}) where {T,N,S} = _unsafe_copy!(KnetArray{T}(undef,size(x)), 1, convert(Array{T,N},x), 1, length(x))

# KnetArray(::AbstractArray)
KnetArray(A::AbstractArray{T,N})    where {T,N}   = KnetArray{T,N}(A)
KnetArray{T}(A::AbstractArray{S,N}) where {T,N,S} = KnetArray{T,N}(A)
KnetArray{T,N}(x::AbstractArray{S,N}) where {T,N,S} = _unsafe_copy!(KnetArray{T}(undef,size(x)), 1, convert(Array{T,N},x), 1, length(x))

# Array(::KnetArray)
import Base: Array
Array(A::KnetArray{T,N})    where {T,N}   = Array{T,N}(A)
Array{T}(A::KnetArray{S,N}) where {T,N,S} = Array{T,N}(A)
Array{T,N}(x::KnetArray{S,N}) where {T,N,S} = convert(Array{T,N}, _unsafe_copy!(Array{S}(undef,size(x)), 1, x, 1, length(x)))

# Conversions:
import Base: convert
# KnetArray <- KnetArray
convert(::Type{KnetArray}, x::KnetArray{T,N}) where {T,N} = x
convert(::Type{KnetArray{T}}, x::KnetArray{T,N}) where {T,N} = x
convert(::Type{KnetArray{T,N}}, x::KnetArray{T,N}) where {T,N} = x
convert(::Type{KnetArray{T}}, x::KnetArray{S,N}) where {T,N,S} = convert(KnetArray{T,N}, x)
convert(::Type{KnetArray{T,N}}, x::KnetArray{S,N}) where {T,N,S} = convert(KnetArray{T,N},_unsafe_copy!(Array{S,N}(undef,size(x)), 1, x, 1, length(x)))

# KnetArray <- AbstractArray
convert(::Type{KnetArray}, x::AbstractArray{T,N}) where {T,N} = convert(KnetArray{T,N}, x)
convert(::Type{KnetArray{T}}, x::AbstractArray{S,N}) where {T,N,S} = convert(KnetArray{T,N}, x)
convert(::Type{KnetArray{T,N}}, x::AbstractArray{S,N}) where {T,N,S} = _unsafe_copy!(KnetArray{T,N}(undef,size(x)), 1, convert(Array{T,N},x), 1, length(x))

# Array <- KnetArray
convert(::Type{Array}, x::KnetArray{T,N}) where {T,N} = convert(Array{T,N}, x)
convert(::Type{Array{T}}, x::KnetArray{S,N}) where {T,N,S} = convert(Array{T,N}, x)
convert(::Type{Array{T,N}}, x::KnetArray{S,N}) where {T,N,S} = convert(Array{T,N},_unsafe_copy!(Array{S}(undef,size(x)), 1, x, 1, length(x)))

# Ptr <- KnetArray
import Base: unsafe_convert, pointer
unsafe_convert(::Type{Ptr{T}}, a::KnetArray) where {T} = unsafe_convert(Ptr{T}, pointer(a))
pointer(a::KnetArray{T}) where {T} = convert(Ptr{T}, a.ptr.ptr)
pointer(a::KnetArray{T},i) where {T} = convert(Ptr{T}, a.ptr.ptr + (i-1)*sizeof(T))

# Reshape:
import Base: reshape, vec
function reshape(a::KnetArray{T}, dims::Dims) where T
    if dims==size(a) 
        a
    elseif prod(dims) != length(a) 
        throw(DimensionMismatch())
    else
        KnetArray{T,length(dims)}(a.ptr, dims)
    end
end

reshape(a::KnetArray, dims::Union{Int,Colon}...) = reshape(a, dims)
reshape(a::KnetArray, dims::Tuple{Vararg{Union{Int,Colon}}}) = reshape(a, Base._reshape_uncolon(a, dims))

vec(a::KnetArray) = reshape(a, length(a))

if isdefined(AutoGrad,:Arg); @eval begin  # TODO: deprecate in next AutoGrad version.
    using AutoGrad: Arg
end; end

# AbstractArray interface
import Base: eachindex, eltype, lastindex, fill!, first, isempty, length, ndims, one, ones, similar, size, stride, strides, zero, (==), isapprox #, linearindexing
eachindex(a::KnetArray) = (1:length(a))
eltype(::KnetArray{T}) where {T}=T
eltype(::Type{KnetArray{T}}) where {T} = T
eltype(::Type{KnetArray{T,n}}) where {T,n} = T
lastindex(a::KnetArray) = length(a)
lastindex(a::KnetArray,d) = size(a,d)
fill!(a::KnetArray{T},x) where {T}=(a[:] .= T(x);a)
first(a::KnetArray) = a[1]
# AutoGrad leaves `first` as a compound proc calling start which doesn't work with KnetArrays
@primitive  first(x::KnetArray),dy,y  AutoGrad.ungetindex(x,dy,1)
isempty(a::KnetArray) = (0==length(a))
length(a::KnetArray)=prod(size(a))
# linearindexing(::KnetArray)=Base.LinearFast() # deprecated in Julia6
ndims(a::KnetArray{T,N}) where {T,N}=N
ones(a::KnetArray{T}) where {T}=fill!(similar(a),one(T))
similar(a::KnetArray, T, dims::Dims)      = KnetArray{T}(undef,dims)
similar(a::KnetArray, T, dims::Int...)    = similar(a, T, dims)
similar(a::KnetArray, T)                  = similar(a, T, size(a))
similar(a::KnetArray{T}) where {T}               = similar(a, T, size(a))
similar(a::KnetArray{T}, dims::Dims) where {T}   = similar(a, T, dims)
similar(a::KnetArray{T}, dims::Int...) where {T} = similar(a, T, dims)
size(a::KnetArray)=a.dims
size(a::KnetArray{T,N},i::Integer) where {T,N}=(if i>N; 1; else; size(a)[i]; end)
stride(a::KnetArray{T,N},i::Integer) where {T,N}=(if i>N; length(a); else; s=1; for n=1:(i-1); s*=size(a,n); end; s; end)
strides(a::KnetArray{T,N}) where {T,N}=ntuple(n->stride(a,n), N)
zero(a::KnetArray{T}) where {T}=fill!(similar(a),zero(T))

# Comparisons
(==)(a::KnetArray{T},b::KnetArray{T}) where {T}=(size(a)==size(b) && norm(a-b)==0)
(==)(a::AbstractArray,b::KnetArray)=(size(a)==size(b) && a==Array(b))
(==)(a::KnetArray,b::AbstractArray)=(size(a)==size(b) && Array(a)==b)
# Adapted from base/linalg/generic.jl:589
isapprox(a::KnetArray{T}, b::KnetArray{T}; rtol=sqrt(eps(T)), atol=T(0)) where {T} =(size(a)==size(b) && norm(a-b) <= atol + rtol * max(norm(a), norm(b)))
isapprox(a::AbstractArray,b::KnetArray;o...)=(size(a)==size(b) && isapprox(a,Array(b);o...))
isapprox(a::KnetArray,b::AbstractArray;o...)=(size(a)==size(b) && isapprox(Array(a),b;o...))



# Concatenation:
import Base: hcat, vcat, cat

# Need to extend cat definitions from AutoGrad/src/base/abstractarray.jl:
const NAVK = Union{Number,AbstractArray,Value,KnetArray}
cat(X::NAVK...; dims) = forw(cat,X...;dims=dims)
if isdefined(AutoGrad,:Arg); @eval begin
    AutoGrad.back(::typeof(cat),::Type{Arg{N}},y1::NAVK,y::NAVK,x::NAVK...; dims) where {N}=AutoGrad.uncat(y1,N,dims,x...)
end; else; @eval begin
    AutoGrad.back(::typeof(cat),::Val{N},y1::NAVK,y::NAVK,x::NAVK...; dims) where {N}=AutoGrad.uncat(y1,N,dims,x...)
end; end

# Benchmarks in μs for hcat and vcat: a=rand(1000,1000) v=rand(1000), t=v'
#		cpu	gpu	g->c->g	vkernel
# hcat(a,a)	2350	225	16160
# hcat(a,v)	1230	115	6490
# hcat(v,a)	1220	120	6490
# hcat(v,v)	3.53	12.53	48.49
# vcat(a,a)	2630	10980	16590	665
# vcat(a,t)	1350	10860	6550	338
# vcat(t,a)	1360	10850	6570	338
# vcat(v,v)	2.13	12.33	45.40	13.58

# setindex! methods called by hcat/vcat:
# hcat(v,v): I = (Colon(),1:1) I = (Colon(),2:2)
# vcat(v,v): uses single index
# hcat(m,m): I = (Colon(),1:5) I = (Colon(),6:10)
# vcat(m,m): I = (1:3,Colon()) I = (4:6,Colon())

# based on typed_hcat{T}(::Type{T}, A::AbstractVecOrMat...) in base/abstractarray.jl:996
function hcat(A::KnetVecOrMat{T}...) where {T}
    nargs = length(A)
    nrows = size(A[1], 1)
    ncols = 0
    for j = 1:nargs
        Aj = A[j]
        if size(Aj, 1) != nrows
            throw(ArgumentError("number of rows of each array must match (got $(map(x->size(x,1), A)))"))
        end
        nd = ndims(Aj)
        ncols += (nd==2 ? size(Aj,2) : 1)
    end
    B = similar(A[1], nrows, ncols)
    pos = 1
    for k = 1:nargs
        Ak = A[k]
        n = length(Ak)
        copyto!(B, pos, Ak, 1, n)
        pos += n
    end
    return B
end

function vcat(A::KnetVector{T}...) where {T}
    nargs = length(A)
    nrows = 0
    for a in A
        nrows += length(a)
    end
    B = similar(A[1], nrows)
    pos = 1
    for k = 1:nargs
        Ak = A[k]
        n = length(Ak)
        copyto!(B, pos, Ak, 1, n)
        pos += n
    end
    return B
end

function vcat(A::KnetVecOrMat{T}...) where {T}
    nargs = length(A)
    nrows = sum(a->size(a, 1), A)::Int
    ncols = size(A[1], 2)
    for j = 2:nargs
        if size(A[j], 2) != ncols
            throw(ArgumentError("number of columns of each array must match (got $(map(x->size(x,2), A)))"))
        end
    end
    B = similar(A[1], nrows, ncols)
    pos = 1
    for k = 1:nargs
        Ak = A[k]
        p1 = pos+size(Ak,1)-1
        B[pos:p1, :] = Ak
        pos = p1+1
    end
    return B
end

function cat(a1::KnetVecOrMat{T}, a::KnetVecOrMat{T}...; dims) where {T}
    if     dims==1 || dims==Val(1); vcat(a1, a...)
    elseif dims==2 || dims==Val(2); hcat(a1, a...)
    else error("cat(a...;dims=$dims) not implemented.")
    end
end

# Avoid using Base for unimplemented cat methods:

using AutoGrad: NA # Union{Number,AbstractArray}
const NAK = Union{Number,AbstractArray,KnetArray}
cat(a::NA, as::NA...; dims)=Base._cat(dims, a, as...)
cat(a::NAK, as::NAK...; dims)=throw(MethodError(cat, (a, as...)))
hcat(a::AbstractArray, as::AbstractArray...)=cat(a,as...; dims=2) # ambiguity fix #321
hcat(a::NA, as::NA...)=cat(a,as...; dims=2)
hcat(a::NAK, as::NAK...)=throw(MethodError(hcat, (a, as...)))
vcat(a::AbstractArray, as::AbstractArray...)=cat(a,as...; dims=1) # ambiguity fix #321
vcat(a::NA, as::NA...)=cat(a,as...; dims=1)
vcat(a::NAK, as::NAK...)=throw(MethodError(vcat, (a, as...)))

# Ambiguity fix for abstractarray.jl:1066-1072
using Base: hvcat_fill, promote_typeof
vcat(X::Number, Xs::Number...) = hvcat_fill(Array{promote_typeof(X, Xs...)}(undef,1+length(Xs)), (X, Xs...))
hcat(X::Number, Xs::Number...) = hvcat_fill(Array{promote_typeof(X, Xs...)}(undef,1,1+length(Xs)), (X, Xs...))

# Utilities:

# Generalizing low level copy using linear indexing to/from gpu arrays:
# copyto!{T}(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer)
# Note that this is an unsafe operation, no argument or bounds checking performed.
# Defined in Base:
# _unsafe_copy!{T}(dest::Ptr{T}, src::Ptr{T}, n) at array.jl:73
# _unsafe_copy!{T}(dest::Array{T,N}, doffs, src::Array{T,N}, soffs, n) at array.jl:79

import Base: copy, copyto! #TODO _unsafe_copy!
const KorA{T} = Union{KnetArray{T},Array{T}}

function copyto!(dest::KorA{T}, doffs::Integer, src::KorA{T}, soffs::Integer, n::Integer) where {T}
    if n == 0; return dest; end
    if n < 0; throw(ArgumentError()); end
    if soffs < 1 || doffs < 1 || soffs+n-1 > length(src) || doffs+n-1 > length(dest)
        throw(BoundsError())
    end
    _unsafe_copy!(dest, doffs, src, soffs, n)
end

function copyto!(dest::KorA{T}, src::KorA{T}) where {T}
    if length(dest) < length(src); throw(BoundsError()); end
    copyto!(dest, 1, src, 1, length(src))
end

function copy(a::KnetArray)
    _unsafe_copy!(similar(a),1,a,1,length(a))
end

# _unsafe_copy! does no bounds checking, the callers must.
function _unsafe_copy!(dest::KnetArray{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int) where {T}
    @cudart(cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
          pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), 1)
    return dest
end
function _unsafe_copy!(dest::Array{T}, doffs::Int, src::KnetArray{T}, soffs::Int, n::Int) where {T}
    @cudart(cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
          pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), 2)
    return dest
end
function _unsafe_copy!(dest::KnetArray{T}, doffs::Int, src::KnetArray{T}, soffs::Int, n::Int) where {T}
    @cudart(cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
          pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), 3)
    return dest
end

# This will make deepcopy work properly
Base.deepcopy_internal(x::KnetArray, s::IdDict)=if haskey(s,x); s[x]; else; copy(x); end

function cudadir(a,b)
    deva = isa(a,KnetArray) && a.ptr.dev >= 0
    devb = isa(b,KnetArray) && b.ptr.dev >= 0
    if !deva && !devb; return 0
    elseif deva && !devb; return 1
    elseif !deva && devb; return 2
    elseif deva && devb;  return 3
    end
end

# Array/KnetArray Transfer

# This works but unnecessarily defines new functions:
# cpu2gpu(x::Array)=KnetArray(x)
# @primitive cpu2gpu(x),dy,y (gpu2cpu(dy))
# gpu2cpu(x::KnetArray)=Array(x)
# @primitive gpu2cpu(x),dy,y (cpu2gpu(dy))

# This does not work because !isa(Array,Function)
# @primitive  KnetArray(x::Array),dy  Array(dy)
# @primitive  Array(x::KnetArray),dy  KnetArray(dy)

# This does not work, parametric methods not yet supported, also unnecessary first arg gradient.
# @primitive convert{A<:AbstractArray,K<:KnetArray}(T::Type{K}, x::Value{A}),dy 0 Array(dy)
# @primitive convert{A<:AbstractArray,K<:KnetArray}(T::Type{A}, x::Value{K}),dy 0 KnetArray(dy)

# So we will define gradients for convert, KnetArray, Array manually:
Base.Array(x::Value{K}) where {K<:KnetArray}=convert(Array,x)
KnetArray(x::Value{A}) where {A<:AbstractArray}=convert(KnetArray,x)
convert(::Type{A},x::Value{K}) where {A<:AbstractArray,K<:KnetArray}=forw(convert,A,x)
convert(::Type{K},x::Value{A}) where {A<:AbstractArray,K<:KnetArray}=forw(convert,K,x)
if isdefined(AutoGrad,:Arg); @eval begin
    AutoGrad.back(::typeof(convert),::Type{Arg{2}},dy,y,T,x) = convert(typeof(value(x)),dy)
end; else; @eval begin
    AutoGrad.back(::typeof(convert),::Val{2},dy,y,T,x) = convert(typeof(value(x)),dy)
end; end

# This gives ambiguity errors:
# @primitive convert(t::Type,x::KnetArray),dy  nothing  convert(KnetArray,dy)
# @primitive convert(t::Type{KnetArray},x::AbstractArray),dy  nothing  convert(Array,dy)

### INDEXING
## Indexing with Real
## Indexing with Tuple{Real}
## Indexing with CartesianIndex: calls Tuple{Real}
## Indexing with AbstractUnitRange
## Indexing with Colon
## Indexing with KnetArray{Int32}: low level, only Int32 supported, no bound checking
## Indexing with (Colon,KnetArray{Int32})
## Indexing with (KnetArray{Int32},Colon)
## Indexing with AbstractArray{Real} calls KnetArray{Int32} after boundchecking
## Indexing with AbstractArray{CartesianIndex} calls AbstractArray{Real}
## Indexing with Empty Array or other unrecognized AbstractArray calls AbstractArray{Real}
## Indexing with (Colon,AbstractVector{Real}) calls (Colon,KnetArray{Int32}) after bound checking
## Indexing with (AbstractVector{Real},Colon) calls (KnetArray{Int32},Colon) after bound checking
## Indexing with StepRange calls AbstractArray{Real}
## Indexing with (StepRange,Colon) calls (AbstractArray{Real},Colon)
## Indexing with (Colon,StepRange) calls (Colon,AbstractArray{Real})
## Indexing with AbstractArray{Bool} calls KnetArray{Int32}; no need for bound checking
## Indexing with (Colon,AbstractVector{Bool}) calls (Colon,KnetArray{Int32}); no need for bound checking
## Indexing with (AbstractVector{Bool},Colon) calls (KnetArray{Int32},Colon); no need for bound checking
## Indexing with KnetArray{T} for logicals calls KnetArray{Int32}
## Indexing with Pair{Union{Real,AbstractUnitRange,Colon}}


import Base: getindex, setindex!, unsafe_getindex, unsafe_setindex!

## Indexing with Real

function getindex(A::KnetArray{T}, I::Real) where {T}
    J = Int(I)
    if !(1 <= J <= length(A)); throw(BoundsError(A,J)); end
    _unsafe_copy!(T[0], 1, A, J, 1)[1]
end

function setindex!(A::KnetArray{T}, v, I::Real) where {T}
    J = Int(I)
    if !(1 <= J <= length(A)); throw(BoundsError(A,J)); end
    _unsafe_copy!(A, J, T[v], 1, 1)
end

## Indexing with Tuple{Real}
# Julia #14770
# If I is shorter than ndims(A) but longer than 1 the remaining indices assumed =1
# Also extra 1's at the end of I are ignored

function getindex(A::KnetArray{T}, I::Real...) where {T}
    J = ntuple(i->Int(I[i]), length(I)) # deprecated: Base.to_indexes(I...)
    @inbounds for j=1:length(J)
        if !(1 <= J[j] <= size(A,j)); throw(BoundsError(A,J)); end
    end
    i = (LinearIndices(size(A)))[J...]
    _unsafe_copy!(T[0], 1, A, i, 1)[1]
end

function setindex!(A::KnetArray{T}, v, I::Real...) where {T}
    J = ntuple(i->Int(I[i]), length(I)) # deprecated: Base.to_indexes(I...)
    @inbounds for j=1:length(J)
        if !(1 <= J[j] <= size(A,j)); throw(BoundsError(A,J)); end
    end
    i = (LinearIndices(size(A)))[J...]
    _unsafe_copy!(A, i, T[v], 1, 1)
end

## Indexing with CartesianIndex: calls Tuple{Real}

function getindex(A::KnetArray{T}, c::CartesianIndex) where {T}
    getindex(A, c.I...)
end

function setindex!(A::KnetArray{T}, v, c::CartesianIndex) where {T}
    setindex!(A, v, c.I...)
end

## Indexing with AbstractUnitRange
# We will implement indexing ranges as views not copies, if possible (when contiguous).
# For contiguous memory without stride all but the last >1 dimension must be full
# The original getindex(a,i:j...) for AbstractArray copies:
# function _getindex(l::LinearIndexing, A::AbstractArray, I::Union{Real, AbstractArray, Colon}...)
# in abstractarray.jl:487,multidimensional.jl:184.

function getindex(A::KnetArray{T}, I::AbstractUnitRange) where {T}
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    off = 1+(first(I)-1)*sizeof(T)
    len = length(I)*sizeof(T)
    ptr = KnetPtr(A.ptr, off, len)
    KnetArray{T,1}(ptr, (length(I),))
end

# Efficient fill:
for S in (32,64); T = Symbol("Float$S"); F = "fill_$S"
    @eval function unsafe_setindex!(a::KnetArray{$T},v::$T,I::AbstractUnitRange)
        @knet8($F,(Cint,$T,Ptr{$T}),length(I),v,pointer(a,first(I)))
    end
end

function setindex!(A::KnetArray{T}, v::Real, I::AbstractUnitRange) where {T}
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    if length(I)==0; return A; end
    unsafe_setindex!(A,T(v),I)
end

function setindex!(A::KnetArray{T}, v::Real, I::AbstractUnitRange{Bool}) where {T} # julia4 ambig fix
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    if length(I)==0; return A; end
    unsafe_setindex!(A,T(v),I)
end

function setindex!(A::KnetArray{T}, v, I::AbstractUnitRange) where {T}
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    if length(v)!=length(I); throw(DimensionMismatch()); end
    if length(I)==0; return A; end
    if eltype(v)!=T; v = convert(Array{T},v); end
    _unsafe_copy!(A,first(I),v,1,length(I))
end

## Indexing with Colon
# Note that getindex(a,:) returns a view not a copy

function getindex(A::KnetArray, I::Colon)
    reshape(A,length(A))
end

function setindex!(A::KnetArray{T}, v::Real, I::Colon) where {T}
    if length(A)==0; return A; end
    unsafe_setindex!(A, T(v), 1:length(A))
end

function setindex!(A::KnetArray{T}, v, I::Colon) where {T}
    if length(v)!=length(A); throw(DimensionMismatch()); end
    if length(v)==0; return A; end
    if eltype(v)!=T; v = convert(Array{T},v); end
    _unsafe_copy!(A,1,v,1,length(A))
end

for F in (32,64); T=Symbol("Float$F"); @eval begin

## Indexing with KnetArray{Int32}: low level, only Int32 supported, no bounds checking

    function unsafe_getindex!(x::KnetArray{$T}, y::KnetArray{$T}, i::KnetArray{Int32})
        @knet8($("getents_$F"),(Cint,Ptr{Int},Ptr{$T},Ptr{$T}), length(i), i, x, y)
    end

    function unsafe_setindex!(x::KnetArray{$T}, y::$T, i::KnetArray{Int32})
        @knet8($("setent1_$F"),(Cint,Ptr{Int},Ptr{$T},$T), length(i), i, x, y)
    end

    function unsafe_setindex!(x::KnetArray{$T}, y::KnetArray{$T}, i::KnetArray{Int32})
        @knet8($("setents_$F"),(Cint,Ptr{Int},Ptr{$T},Ptr{$T}), length(i), i, x, y)
    end

## Indexing with (Colon,KnetArray{Int32})
# TODO: Just special case rows and columns in matrices until we have a more general solution

    function unsafe_getindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, ::Colon, i::KnetVector{Int32})
        @knet8($("getcols_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(x,1), size(x,2), length(i), i, x, y)
    end

    function unsafe_setindex!(x::KnetMatrix{$T}, y::$T, ::Colon, i::KnetVector{Int32})
        @knet8($("setcol1_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},$T),
               size(x,1), size(x,2), length(i), i, x, y)
    end

    function unsafe_setindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, ::Colon, i::KnetVector{Int32})
        @knet8($("setcols_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(x,1), size(x,2), length(i), i, x, y)
    end

## Indexing with (KnetArray{Int32},Colon)

    function unsafe_getindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, i::KnetVector{Int32}, ::Colon)
        @knet8($("getrows_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(x,1), size(x,2), length(i), i, x, y)
    end

    function unsafe_setindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, i::KnetVector{Int32}, ::Colon)
        @knet8($("setrows_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(x,1), size(x,2), length(i), i, x, y)
    end

    function unsafe_setindex!(x::KnetMatrix{$T}, y::$T, i::KnetVector{Int32}, ::Colon)
        @knet8($("setrow1_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},$T),
               size(x,1), size(x,2), length(i), i, x, y)
    end

end; end

# bound checking

function checkbetween(i::AbstractArray{I},lo::L,hi::H) where {I<:Integer,L<:Integer,H<:Integer}
    checkbetween(Array{Int32}(i),Int32(lo),Int32(hi))
end

function checkbetween(i::Array{Int32},lo::Int32,hi::Int32)
    @inbounds for ii in i
        if !(lo <= ii <= hi)
            throw(BoundsError(lo:hi, ii))
        end
    end
end

## Indexing with AbstractArray{Real} calls KnetArray{Int32} after boundchecking

function getindex(x::KnetArray{T}, i::AbstractArray{I}) where {T,I<:Real}
    y = similar(x, size(i))
    if isempty(y); return y; end
    i = Array{Int32}(i)
    checkbetween(i, 1, length(x))
    unsafe_getindex!(x,y,KnetArray{Int32}(i))
    return y
end

function setindex!(x::KnetArray{T}, y::Real, i::AbstractArray{I}) where {T,I<:Real}
    if isempty(i); return x; end
    i = Array{Int32}(i)
    checkbetween(i, 1, length(x))
    unsafe_setindex!(x,T(y),KnetArray{Int32}(i))
    return x
end

function setindex!(x::KnetArray{T}, y, i::AbstractArray{I}) where {T,I<:Real}
    if length(y) != length(i); throw(DimensionMismatch()); end
    if isempty(i); return x; end
    i = Array{Int32}(i)
    checkbetween(i, 1, length(x))
    unsafe_setindex!(x,KnetArray{T}(y),KnetArray{Int32}(i))
    return x
end

## Indexing with (Colon,AbstractVector{Real}) calls (Colon,KnetArray{Int32}) after bound checking

function getindex(x::KnetMatrix{T}, c::Colon, i::AbstractVector{I}) where {T,I<:Real}
    xrows,xcols = size(x); ycols = length(i)
    y = similar(x, xrows, ycols)
    if isempty(y); return y; end
    i = Array{Int32}(i)
    checkbetween(i,1,xcols)
    unsafe_getindex!(x,y,c,KnetArray{Int32}(i))
    return y
end

function setindex!(x::KnetMatrix{T}, y::Real, c::Colon, i::AbstractVector{I}) where {T,I<:Real}
    if isempty(i); return x; end
    xrows,xcols = size(x); ycols=length(i)
    i = Array{Int32}(i)
    checkbetween(i,1,xcols)
    unsafe_setindex!(x,T(y),c,KnetArray{Int32}(i))
    return x
end

function setindex!(x::KnetMatrix{T}, y, c::Colon, i::AbstractVector{I}) where {T,I<:Real}
    if ndims(y) != 2; throw(DimensionMismatch()); end
    xrows,xcols = size(x); yrows,ycols=size(y)
    if yrows != xrows; throw(DimensionMismatch()); end
    if ycols != length(i); throw(DimensionMismatch()); end
    if isempty(y); return x; end
    i = Array{Int32}(i)
    checkbetween(i,1,xcols)
    unsafe_setindex!(x,KnetArray{T}(y),c,KnetArray{Int32}(i))
    return x
end

function setindex!(x::KnetMatrix{T}, y, c::AbstractUnitRange, i::AbstractVector{I}) where {T,I<:Real}
    if c == 1:size(x,1)
        setindex!(x, y, :, i)
    else
        throw(MethodError(setindex!,x,y,c,i))
    end
end

## Indexing with (AbstractVector{Real},Colon) calls (KnetArray{Int32},Colon) after bound checking

function getindex(x::KnetMatrix{T}, i::AbstractVector{I}, c::Colon) where {T,I<:Real}
    xrows,xcols = size(x); yrows = length(i)
    y = similar(x, yrows, xcols)
    if isempty(y); return y; end
    i = Array{Int32}(i)
    checkbetween(i,1,xrows)
    unsafe_getindex!(x,y,KnetArray{Int32}(i),c)
    return y
end

function setindex!(x::KnetMatrix{T}, y::Real, i::AbstractVector{I}, c::Colon) where {T,I<:Real}
    if isempty(i); return x; end
    xrows,xcols = size(x); yrows=length(i)
    i = Array{Int32}(i)
    checkbetween(i,1,xrows)
    unsafe_setindex!(x,T(y),KnetArray{Int32}(i),c)
    return x
end

function setindex!(x::KnetMatrix{T}, y, i::AbstractVector{I}, c::Colon) where {T,I<:Real}
    if ndims(y) != 2; throw(DimensionMismatch()); end
    xrows,xcols = size(x); yrows,ycols=size(y)
    if ycols != xcols; throw(DimensionMismatch()); end
    if yrows != length(i); throw(DimensionMismatch()); end
    if isempty(y); return x; end
    i = Array{Int32}(i)
    checkbetween(i,1,xrows)
    unsafe_setindex!(x,KnetArray{T}(y),KnetArray{Int32}(i),c)
    return x
end

function setindex!(x::KnetMatrix{T}, y, i::AbstractVector{I}, c::AbstractUnitRange) where {T,I<:Real}
    if c == 1:size(x,2)
        setindex!(x, y, i, :)
    else
        throw(MethodError(setindex!,x,y,i,c))
    end
end

## Indexing with AbstractArray{CartesianIndex} calls AbstractArray{Real}

c2i(d::Dims,i::AbstractArray{I}) where {I<:CartesianIndex} = Int32[(LinearIndices(d))[c.I...] for c in i]
getindex(x::KnetArray{T}, i::AbstractArray{I}) where {T,I<:CartesianIndex} = getindex(x, c2i(size(x),i))
setindex!(x::KnetArray{T}, y, i::AbstractArray{I}) where {T,I<:CartesianIndex} = setindex!(x, y, c2i(size(x),i))

## Indexing with Empty Array or other unrecognized AbstractArray calls AbstractArray{Real}

getindex(x::KnetArray{T}, i::AbstractArray) where {T}=getindex(x, Array{Int32}(i))
setindex!(x::KnetArray{T}, y, i::AbstractArray) where {T}=setindex!(x, y, Array{Int32}(i))

## Indexing with StepRange calls AbstractArray{Real}

function getindex(A::KnetArray{T}, I::StepRange) where {T}
    getindex(A, collect(I))
end

function setindex!(A::KnetArray{T}, v::Real, I::StepRange{R}) where {T,R<:Real} # julia4 ambiguity fix
    setindex!(A, v, collect(I))
end

function setindex!(A::KnetArray{T}, v::Real, I::StepRange{Bool}) where {T} # julia4 ambiguity fix
    setindex!(A, v, collect(I))
end

function setindex!(A::KnetArray{T}, v, I::StepRange) where {T}
    setindex!(A, v, collect(I))
end

## Indexing with (StepRange,Colon) calls (AbstractArray{Real},Colon)

function getindex(A::KnetMatrix{T}, I::StepRange, c::Colon) where {T}
    getindex(A, collect(I), c)
end

function setindex!(A::KnetMatrix{T}, v::Real, I::StepRange{R}, c::Colon) where {T,R<:Real} # julia4 ambig fix
    setindex!(A, v, collect(I), c)
end

function setindex!(A::KnetMatrix{T}, v::Real, I::StepRange{Bool}, c::Colon) where {T} # julia4 ambig fix
    setindex!(A, v, collect(I), c)
end

function setindex!(A::KnetMatrix{T}, v, I::StepRange, c::Colon) where {T}
    setindex!(A, v, collect(I), c)
end

function setindex!(A::KnetMatrix, v, I::StepRange, r::AbstractUnitRange)
    if r == 1:size(A,2)
        setindex!(A,v,I,:)
    else
        throw(MethodError(setindex!, A, v, I, r))
    end
end

## Indexing with (Colon,StepRange) calls (Colon,AbstractArray{Real})

function getindex(A::KnetMatrix{T}, c::Colon, I::StepRange) where {T}
    getindex(A, c, collect(I))
end

function setindex!(A::KnetMatrix{T}, v::Real, c::Colon, I::StepRange{R}) where {T,R<:Real} # julia4 ambig fix
    setindex!(A, v, c, collect(I))
end

function setindex!(A::KnetMatrix{T}, v::Real, c::Colon, I::StepRange{Bool}) where {T} # julia4 ambig fix
    setindex!(A, v, c, collect(I))
end

function setindex!(A::KnetMatrix{T}, v, c::Colon, I::StepRange) where {T}
    setindex!(A, v, c, collect(I))
end

function setindex!(A::KnetMatrix, v, r::AbstractUnitRange, I::StepRange)
    if r == 1:size(A,1)
        setindex!(A,v,:,I)
    else
        throw(MethodError(setindex!, A, v, r, I))
    end
end

## Indexing with AbstractArray{Bool} calls KnetArray{Int32}; no need for bound checking

function getindex(x::KnetArray{T}, i::AbstractArray{Bool}) where {T}
    if length(i) != length(x); throw(BoundsError(x,i)); end
    j = findall(vec(i))
    y = similar(x, length(j))
    if !isempty(y); unsafe_getindex!(x,y,KnetArray{Int32}(j)); end
    return y
end

function setindex!(x::KnetArray{T}, y::Real, i::AbstractArray{Bool}) where {T}
    if length(i) != length(x); throw(DimensionMismatch()); end
    j = findall(vec(i))
    if !isempty(j); unsafe_setindex!(x,T(y),KnetArray{Int32}(j)); end
    return x
end

function setindex!(x::KnetArray{T}, y, i::AbstractArray{Bool}) where {T}
    if length(i) != length(x); throw(DimensionMismatch()); end
    j = findall(vec(i))
    if length(j) != length(y); throw(BoundsError(y,j)); end
    if !isempty(j); unsafe_setindex!(x,KnetArray{T}(y),KnetArray{Int32}(j)); end
    return x
end

## Indexing with (Colon,AbstractVector{Bool}) calls (Colon,KnetArray{Int32}); no need for bound checking

function getindex(x::KnetMatrix{T}, c::Colon, i::AbstractVector{Bool}) where {T}
    xrows,xcols = size(x)
    if length(i) != xcols; throw(BoundsError(x,(:,i))); end
    j = findall(vec(i)); ycols = length(j)
    y = similar(x, xrows, ycols)
    if !isempty(y); unsafe_getindex!(x,y,c,KnetArray{Int32}(j)); end
    return y
end

function setindex!(x::KnetMatrix{T}, y::Real, c::Colon, i::AbstractVector{Bool}) where {T}
    xrows,xcols = size(x)
    if length(i) != xcols; throw(BoundsError(x,(:,i))); end
    j = findall(vec(i))
    if !isempty(j); unsafe_setindex!(x,T(y),c,KnetArray{Int32}(j)); end
    return x
end

function setindex!(x::KnetMatrix{T}, y, c::Colon, i::AbstractVector{Bool}) where {T}
    if ndims(y) != 2; throw(DimensionMismatch()); end
    xrows,xcols = size(x); yrows,ycols=size(y)
    if yrows != xrows; throw(DimensionMismatch()); end
    if length(i) != xcols; throw(BoundsError(x,(:,i))); end
    j = findall(vec(i))
    if ycols != length(j); throw(DimensionMismatch()); end
    if !isempty(y); unsafe_setindex!(x,KnetArray{T}(y),c,KnetArray{Int32}(j)); end
    return x
end

function setindex!(x::KnetMatrix, y, c::AbstractUnitRange, i::AbstractVector{Bool})
    if c == 1:size(x,1)
        setindex!(x,y,:,i)
    else
        throw(MethodError(setindex!,x,y,c,i))
    end
end

## Indexing with (AbstractVector{Bool},Colon) calls (KnetArray{Int32},Colon); no need for bound checking

function getindex(x::KnetMatrix{T}, i::AbstractVector{Bool}, c::Colon) where {T}
    xrows,xcols = size(x)
    if length(i) != xrows; throw(BoundsError(x,(i,:))); end
    j = findall(vec(i)); yrows = length(j)
    y = similar(x, yrows, xcols)
    if !isempty(y); unsafe_getindex!(x,y,KnetArray{Int32}(j),c); end
    return y
end

function setindex!(x::KnetMatrix{T}, y::Real, i::AbstractVector{Bool}, c::Colon) where {T}
    xrows,xcols = size(x)
    if length(i) != xrows; throw(BoundsError(x,(i,:))); end
    j = findall(vec(i))
    if !isempty(j); unsafe_setindex!(x,T(y),KnetArray{Int32}(j),c); end
    return x
end

function setindex!(x::KnetMatrix{T}, y, i::AbstractVector{Bool}, c::Colon) where {T}
    if ndims(y) != 2; throw(DimensionMismatch()); end
    xrows,xcols = size(x); yrows,ycols=size(y)
    if ycols != xcols; throw(DimensionMismatch()); end
    if length(i) != xrows; throw(BoundsError(x,(i,:))); end
    j = findall(vec(i))
    if yrows != length(j); throw(DimensionMismatch()); end
    if !isempty(y); unsafe_setindex!(x,KnetArray{T}(y),KnetArray{Int32}(j),c); end
    return x
end

function setindex!(x::KnetMatrix, y, i::AbstractVector{Bool}, c::AbstractUnitRange)
    if c == 1:size(x,2)
        setindex!(x,y,i,:)
    else
        throw(MethodError(setindex!,x,y,i,c))
    end
end

## Indexing with KnetArray{T} for logicals calls KnetArray{Int32}
# Need this because (k.<0) returns KnetArray{T} instead of BitArray

function getindex(x::KnetArray{T}, i::KnetArray{T}) where {T}
    if length(i) != length(x); throw(BoundsError(x,i)); end
    j = findall(Array(i))
    y = similar(x, length(j))
    if !isempty(y); unsafe_getindex!(x,y,KnetArray{Int32}(j)); end
    return y
end

function setindex!(x::KnetArray{T}, y::Real, i::KnetArray{T}) where {T}
    if length(i) != length(x); throw(DimensionMismatch()); end
    j = findall(Array(i))
    if !isempty(j); unsafe_setindex!(x,T(y),KnetArray{Int32}(j)); end
    return x
end

function setindex!(x::KnetArray{T}, y, i::KnetArray{T}) where {T}
    if length(i) != length(x); throw(DimensionMismatch()); end
    j = findall(Array(i))
    if length(j) != length(y); throw(BoundsError(y,j)); end
    if !isempty(j); unsafe_setindex!(x,KnetArray{T}(y),KnetArray{Int32}(j)); end
    return x
end

# To avoid ambiguity with previous definitions we have these
# TODO: clean this up...
getindex(A::KnetMatrix, I1::Colon, I2::Colon)=getindex2(A,I1,I2)
setindex!(A::KnetMatrix, B, I1::Colon, I2::Colon)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Number, I1::Colon, I2::Colon)=setindex2!(A,B,I1,I2)
getindex(A::KnetMatrix, I1::AbstractUnitRange, I2::AbstractUnitRange)=getindex2(A,I1,I2)
setindex!(A::KnetMatrix, B, I1::AbstractUnitRange, I2::AbstractUnitRange)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Number, I1::AbstractUnitRange, I2::AbstractUnitRange)=setindex2!(A,B,I1,I2)
getindex(A::KnetMatrix, I1::Colon, I2::AbstractUnitRange)=getindex2(A,I1,I2)
setindex!(A::KnetMatrix, B::Real, I1::Colon, I2::AbstractUnitRange{Bool})=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Real, I1::Colon, I2::AbstractUnitRange{T}) where {T<:Real}=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B, I1::Colon, I2::AbstractUnitRange)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Number, I1::Colon, I2::AbstractUnitRange)=setindex2!(A,B,I1,I2)
getindex(A::KnetMatrix, I1::AbstractUnitRange, I2::Colon)=getindex2(A,I1,I2)
setindex!(A::KnetMatrix, B::Real, I1::AbstractUnitRange{Bool}, I2::Colon)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Real, I1::AbstractUnitRange{R}, I2::Colon) where {R<:Real}=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B, I1::AbstractUnitRange, I2::Colon)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Number, I1::AbstractUnitRange, I2::Colon)=setindex2!(A,B,I1,I2)
getindex(A::KnetMatrix, I1::Real, I2::AbstractUnitRange)=getindex2(A,I1,I2)
setindex!(A::KnetMatrix, B, I1::Real, I2::AbstractUnitRange)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Number, I1::Real, I2::AbstractUnitRange)=setindex2!(A,B,I1,I2)
getindex(A::KnetMatrix, I1::AbstractUnitRange, I2::Real)=getindex2(A,I1,I2)
setindex!(A::KnetMatrix, B, I1::AbstractUnitRange, I2::Real)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Number, I1::AbstractUnitRange, I2::Real)=setindex2!(A,B,I1,I2)
getindex(A::KnetMatrix, I1::Colon, I2::Real)=getindex2(A,I1,I2)
setindex!(A::KnetMatrix, B, I1::Colon, I2::Real)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Number, I1::Colon, I2::Real)=setindex2!(A,B,I1,I2)
getindex(A::KnetMatrix, I1::Real, I2::Colon)=getindex2(A,I1,I2)
setindex!(A::KnetMatrix, B, I1::Real, I2::Colon)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Number, I1::Real, I2::Colon)=setindex2!(A,B,I1,I2)

## Indexing with Pair{Union{Real,AbstractUnitRange,Colon}}
# TODO: the following getindex, setindex! work for 1 and 2 dimensions only, write general versions.

const Index3 = Union{Real,AbstractUnitRange,Colon}

function getindex2(A::KnetMatrix{T}, I1::Index3, I2::Index3) where {T}
    (nelts,nrows,ncols,firstindex,astep) = indexparams(A,I1,I2)
    B1 = isa(I1,Colon) ? size(A,1) : length(I1)
    B2 = isa(I2,Colon) ? size(A,2) : length(I2)
    Bsize = isa(I1,Real) ? (B2,) : isa(I2,Real) ? (B1,) : (B1,B2)
    Bdims = length(Bsize)
    if ncols == 1
        off = 1+(firstindex-1)*sizeof(T)
        len = nrows*sizeof(T)
        ptr = KnetPtr(A.ptr, off, len)
        KnetArray{T,Bdims}(ptr, Bsize)
    else
        B = similar(A, Bsize)
        if isempty(B); return B; end
        nrows *= sizeof(T); astep *= sizeof(T)
        @knet8(xcopy,(Cint,Cint,Cptr,Cint,Cptr,Cint),
               nrows, ncols, pointer(A,firstindex), astep, B, nrows)
        return B
    end
end

function setindex2!(A::KnetMatrix{T}, B, I1::Index3, I2::Index3) where {T}
    (nelts,nrows,ncols,firstindex,astep) = indexparams(A,I1,I2)
    aptr0 = pointer(A, firstindex)
    if isa(B,Number)
        B = T(B)
        if ncols == 1
            if nelts > 0
                if T <: Float32
                    @knet8(fill_32,(Cint,Cfloat, Ptr{Cfloat}), nelts,B,aptr0)
                elseif T<: Float64
                    @knet8(fill_64,(Cint,Cdouble,Ptr{Cdouble}),nelts,B,aptr0)
                else
                    error("$T not supported")
                end
            end
        elseif nrows > 0 && ncols > 0
            if T <: Float32
                @knet8(xfill_32,(Cint,Cint,Cfloat, Ptr{Cfloat}, Cint),nrows,ncols,B,aptr0,astep)
            elseif T<: Float64
                @knet8(xfill_64,(Cint,Cint,Cdouble,Ptr{Cdouble},Cint),nrows,ncols,B,aptr0,astep)
            else
                error("$T not supported")
            end
        end
    else
        length(B) == nelts || throw(DimensionMismatch())
        B = convert(KnetArray{T},B)
        if ncols == 1
            if nelts > 0
                @cudart(cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
                      aptr0, B, nelts*sizeof(T), cudadir(A,B))
            end
        elseif nrows > 0 && ncols > 0
            nrows *= sizeof(T); astep *= sizeof(T)
            @knet8(xcopy,(Cint,Cint,Cptr,Cint,Cptr,Cint), nrows, ncols, B, nrows, aptr0, astep)
        end
    end
    return A
end

function indexparams(A::KnetArray{T,N}, I::Index3...) where {T,N}
    N > 2 && error("setindex for ndims > 2 not implemented yet")
    skipped = false
    nrows = nelts = 1
    subs1 = ones(Int,2)
    astep = length(A)
    for i=1:length(I)
        Ii = I[i]
        Ai = size(A,i)
        if isa(Ii, Colon)
            Li = Ai
            subs1[i] = 1
        elseif isa(Ii, Real)
            1 <= Ii <= Ai || throw(DimensionMismatch())
            Li = 1
            subs1[i] = Int(Ii)
        else
            1 <= first(Ii) <= last(Ii) <= Ai || throw(DimensionMismatch())
            Li = length(Ii)
            subs1[i] = first(Ii)
        end
        nelts *= Li
        if !skipped
            nrows *= Li
            if Li < Ai
                skipped = true
                astep = stride(A,i+1)
            end
        end
    end
    ncols = div(nelts, nrows)
    firstindex = (LinearIndices(size(A)))[subs1...]
    return (nelts,nrows,ncols,firstindex,astep)
end


# These two are not sufficient in spite of what the documentation says:
# display goes into an infinite loop!
# getindex{T}(A::KnetArray{T}, i::Int)=_unsafe_copy!(T[0], 1, A, i, 1)[1]
# setindex!{T}(A::KnetArray{T}, v, i::Int)=_unsafe_copy!(A, i, T[v], 1, 1)


# AutoGrad functions:
import AutoGrad: zeroslike, sum_outgrads, UngetIndex # , unary_nd, indexed_function, isequivalent, _dbg, ssize
zeroslike(a::KnetArray)=zero(a)
# unary_nd(f, x::KnetArray, eps) = reshape(eltype(x)[unary_nd(indexed_function(f, x, i), x[i], eps) for i in 1:length(x)], size(x))
# isequivalent(x::Union{KnetArray,AbstractArray}, y::Union{KnetArray,AbstractArray}; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))
# _dbg(a::KnetArray) = "K"*_dbg(Array(a))

# Note that KnetArray sum_outgrads is overwriting, i.e. does not support higher order gradients.
sum_outgrads(a::KnetArray{T},b::KnetArray{T}) where {T}=axpy!(1,b,a) # (a+b)

function sum_outgrads(a::KnetArray,b::UngetIndex)
    c = sum_outgrads_karray(a, b.value, b.index...)
    return c
end

# This only works when there are no repeated indices. This is true for index types:
# Real, (Real...), CartesianIndex, Colon, AbstractArray{Bool}, Range, EmptyArray
# and pairs of Union{Real,AbstractUnitRange,Colon} and (Colon,Range)
sum_outgrads_karray(A::KnetArray, X, I...)=setindex!(A, getindex(A,I...) .+ X, I...)

# The following index types may have repeated indices:
# AbstractArray{Real}, AbstractArray{CartesianIndex}, (Colon,AbstractVector{Real}), (AbstractVector{Real},Colon)

sum_outgrads_karray(A::KnetArray, X, I::AbstractArray{T}) where {T<:CartesianIndex}=sum_outgrads_karray(A,X,c2i(size(A),I))

for F in (32,64); T=Symbol("Float$F"); @eval begin

    function sum_outgrads_karray(A::KnetArray{$T}, X, I::AbstractArray{R}) where {R<:Real}
        I = KnetArray{Int32}(I)
        X = KnetArray{$T}(X)
        @knet8($("addents_$F"),(Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               length(I), I, A, X)
        return A
    end

    function sum_outgrads_karray(A::KnetArray{$T}, X, ::Colon, I::AbstractArray{R}) where {R<:Real}
        I = KnetArray{Int32}(I)
        X = KnetArray{$T}(X)
        @knet8($("addcols_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(A,1), size(A,2), length(I), I, A, X)
        return A
    end

    function sum_outgrads_karray(A::KnetArray{$T}, X, I::AbstractArray{R}, ::Colon) where {R<:Real}
        I = KnetArray{Int32}(I)
        X = KnetArray{$T}(X)
        @knet8($("addrows_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(A,1), size(A,2), length(I), I, A, X)
        return A
    end

    sum_outgrads_karray(A::KnetArray{$T}, X, I::AbstractArray{Bool})=sum_outgrads_karray(A,X,findall(vec(I)))
    sum_outgrads_karray(A::KnetArray{$T}, X, c::Colon, I::AbstractArray{Bool})=sum_outgrads_karray(A,X,c,findall(vec(I)))
    sum_outgrads_karray(A::KnetArray{$T}, X, I::AbstractArray{Bool}, c::Colon)=sum_outgrads_karray(A,X,findall(vec(I)),c)

end; end

# To prevent RSI
ka = KnetArray
export ka


### k[1:2,3:4] .= 0 => materialize!(dotview(k,1:2,3:4),broadcasted(identity,0))

# The following adapted from base: views.jl broadcast.jl subarray.jl
# Much of this will be unnecessary if we can inherit from AbstractArray
import Base: dotview, view, unsafe_view, copyto!, eachindex, _maybe_reshape_parent, reshape, to_shape, SubArray, compute_stride1, axes, check_parent_index_match, IndexStyle
using Base: unalias, index_ndims, @_inline_meta, @boundscheck, ViewIndex, OneTo, rdims, viewindexing, ensure_indexable, index_dimsum, fill_to_length
using Base.Broadcast: Broadcasted

dotview(A::KnetArray,I...) = view(A,I...)

function view(A::KnetArray, I::Vararg{Any,N}) where {N}
    @_inline_meta
    J = map(i->unalias(A,i), to_indices(A, I))
    #TODO: @boundscheck checkbounds(A, J...)
    unsafe_view(_maybe_reshape_parent(A, index_ndims(J...)), J...)
end

function unsafe_view(A::KnetArray, I::Vararg{ViewIndex,N}) where {N}
    @_inline_meta
    SubArray(A, I)
end

eachindex(::IndexLinear, A::KnetArray) = (@_inline_meta; OneTo(length(A)))

_maybe_reshape_parent(A::KnetArray, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::KnetArray{<:Any,1}, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::KnetArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A
_maybe_reshape_parent(A::KnetArray, ::NTuple{N, Bool}) where {N} = reshape(A, Val(N))

reshape(parent::KnetArray{T,N}, ndims::Val{N}) where {T,N} = parent
reshape(parent::KnetArray, ndims::Val{N}) where N = reshape(parent, rdims(Val(N), axes(parent)))
reshape(parent::KnetArray, shp::Tuple{Union{Integer,OneTo}, Vararg{Union{Integer,OneTo}}}) = reshape(parent, to_shape(shp))

function SubArray(parent::KnetArray, indices::Tuple)
    @_inline_meta
    SubArray(IndexStyle(viewindexing(indices), IndexStyle(parent)), parent, ensure_indexable(indices), index_dimsum(indices...))
end

IndexStyle(::KnetArray)=IndexLinear()

compute_stride1(parent::KnetArray, I::NTuple{N,Any}) where {N} =
    (@_inline_meta; compute_stride1(1, fill_to_length(axes(parent), OneTo(1), Val(N)), I))

function axes(A::KnetArray{T,N}, d) where {T,N}
    @_inline_meta
    d <= N ? axes(A)[d] : OneTo(1)
end

check_parent_index_match(parent::KnetArray{T,N}, ::NTuple{N, Bool}) where {T,N} = nothing

# dotview(P::KnetArray{T,N},I...) where {T,N} =SubArray{T,N,typeof(P),typeof(I),false}(P,I,0,0)
# check_parent_index_match(parent::KnetArray{T,N}, ::NTuple{N, Bool}) where {T,N} = nothing
copyto!(a::SubArray{T,N,P,I,L},b::Broadcasted) where {T,N,P<:KnetArray,I,L} = setindex!(a.parent, copy(b), a.indices...)
copyto!(a::SubArray{T,N,P,I,L},b::Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) where {T,N,P<:KnetArray,I,L} = (if !isempty(b); setindex!(a.parent, first(b), a.indices...); end)

# We need x[:,:,t] and hx[:,:,l] for RNNs
function getindex(A::KnetArray, ::Colon, ::Colon, I::Real)
    B = reshape(A, stride(A,3), size(A,3))
    reshape(B[:,I], size(A,1), size(A,2))
end

function getindex(A::KnetArray, ::Colon, ::Colon, ::Colon)
    A
end

function getindex(A::KnetArray, ::Colon, ::Colon, I::Index3)
    B = reshape(A, stride(A,3), size(A,3))
    reshape(B[:,I], size(A,1), size(A,2), length(I))
end

function setindex!(x::KnetArray, y, ::Colon, ::Colon, I::Index3)
    reshape(x, stride(x,3), size(x,3))[:,I] = y
    return x
end

function setindex!(x::KnetArray, y, ::Colon, ::Colon, ::Colon)
    copyto!(x,y)
    return x
end

function setindex!(x::KnetArray, y, i::AbstractUnitRange, j::AbstractUnitRange, k::Index3)
    if first(i) == 1 && last(i) == size(x,1) && first(j) == 1 && last(j) == size(x,2)
        setindex!(x, y, :, :, k)
    else
        throw(MethodError(setindex!, (x,y,i,j,k)))
    end
end

function getindex(x::KnetArray{T,2}, ::Colon, m::Array{I,2}) where {T,I<:Integer}
    reshape(x[:,vec(m)], size(x,1), size(m,1), size(m,2))
end


# https://docs.julialang.org/en/stable/manual/types/#man-custom-pretty-printing-1
# Base.show(io::IO, z): single line format used in show, print, inside other objects.
# Base.show(io::IO, ::MIME"text/plain", z): multi-line format used by display.
# Base.show(io::IO, ::MIME"text/html", z): multi-line format for html output.
# get(io, :compact, false), show(IOContext(stdout, :compact=>true),z) for compact (array) printing.
# summary(io::IO, x) = print(io, typeof(x))
# string(z): uses print_to_string.

import Base: show, summary, display, size, getindex

# Hack for printing without copying the whole KnetArray and without inheriting AbstractArray:
struct KnetDisplay{T,N} <: AbstractArray{T,N}; a::KnetArray{T,N}; end
getindex(a::KnetDisplay, i...) = getindex(a.a, i...)
size(a::KnetDisplay) = size(a.a)
summary(io::IO, a::KnetDisplay) = summary(io, a.a)
summary(io::IO, a::KnetArray) = print(io, Base.dims2string(size(a)), " ", typeof(a))
show(io::IO, a::KnetArray) = (print(io,"K"); show(io, KnetDisplay(a)))
show(io::IO, m::MIME"text/plain", a::KnetArray) = show(io, m, KnetDisplay(a))
summary(io::IO, x::Value{A}) where {A<:KnetArray} = print(io, Base.dims2string(size(x)), " ", typeof(x))


## Broadcasting:

# Both f.(x...) and broadcast(f,x...) turn into materialize(broadcasted(::BroadcastStyle,f,x...)).
import .Broadcast: BroadcastStyle, Style, broadcastable, broadcasted, Broadcasted

# Any call involving KnetArray should be unfused: (see AutoGrad/src/core.notes)
broadcasted(::Style{KnetArray}, f, args...) = f(Bcasted.(args)...).value

# The following should set the style for any call that involves a KnetArray:
BroadcastStyle(::Type{<:KnetArray}) = Style{KnetArray}()
broadcastable(x::KnetArray) = x  # This is necessary for the style stuff to work, default definition `collect(x)` turns x into Array.

# Make sure the KnetArray style overrides others except the AutoGrad.Value style:
BroadcastStyle(k::Style{KnetArray}, s::BroadcastStyle) = k
BroadcastStyle(k::Style{KnetArray}, v::Style{AutoGrad.Value}) = v

# We use a different Bcasted type than AutoGrad to avoid infinite loops:
struct Bcasted{T}; value::T; end

# This fixes (x .- log.(sum(exp.(x),dims=:))) where log.(::Number) gives a Broadcasted object
Bcasted(x::Broadcasted) = Bcasted(copy(x))

# For broadcasting Knet primitives the following needs to be defined (see unary.jl, binary.jl)
# f(x::Bcasted) = broadcasted(f, x.value) |> Bcasted
# broadcasted(f,x::Bcasted) = broadcasted(f, x.value) |> Bcasted
