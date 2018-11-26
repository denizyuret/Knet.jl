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
    ptr::CuArray{T,N}
    dims::NTuple{N,Int}
end

# Note: I removed <: AbstractArray{T,N} after I painfully discovered
# some inefficient AbstractArray methods inherited unintentionally.
# It is better to define a few extra methods to keep a tighter control
# on what methods exactly get called for KnetArrays.

# TODO: Let's see if this keeps it under control:
import Base: getindex, setindex!, iterate, IndexStyle
IndexStyle(::Type{<:KnetArray})=IndexLinear()
# TODO: do we need more defensive methods here?  broadcasted, materialize etc?

# Aliases:

const KnetMatrix{T} = KnetArray{T,2}
const KnetVector{T} = KnetArray{T,1}
const KnetVecOrMat{T} = Union{KnetVector{T}, KnetMatrix{T}}

# Constructors:
# Internal constructor defines KnetArray{T,N}(ptr,dims)

KnetArray(x::CuArray) = KnetArray{eltype(x),ndims(x)}(x, size(x))

# These define KnetArray{T,N}(undef,dims) and KnetArray{T,N}(undef,d...)
KnetArray{T,N}(::UndefInitializer, d::Vararg{Int,N}) where {T,N} = KnetArray{T,N}(CuArray{T}(undef, d), d)
KnetArray{T,N}(::UndefInitializer, d::NTuple{N,Int}) where {T,N} = KnetArray{T,N}(CuArray{T}(undef, d), d)
KnetArray{T,N}(::UndefInitializer, d::Vararg{Integer,N}) where {T,N} = KnetArray{T,N}(CuArray{T}(undef, d), convert(NTuple{N,Int},d))
KnetArray{T,N}(::UndefInitializer, d::NTuple{N,Integer}) where {T,N} = KnetArray{T,N}(CuArray{T}(undef, d), convert(NTuple{N,Int},d))

# These define KnetArray{T}(undef,dims) and KnetArray{T}(undef,d...)
KnetArray{T}(::UndefInitializer, d::Vararg{Int,N}) where {T,N} = KnetArray{T,N}(CuArray{T}(undef, d), d)
KnetArray{T}(::UndefInitializer, d::NTuple{N,Int}) where {T,N} = KnetArray{T,N}(CuArray{T}(undef, d), d)
KnetArray{T}(::UndefInitializer, d::Vararg{Integer,N}) where {T,N} = KnetArray{T,N}(CuArray{T}(undef, Int.(d)), convert(NTuple{N,Int},d))
KnetArray{T}(::UndefInitializer, d::NTuple{N,Integer}) where {T,N} = KnetArray{T,N}(CuArray{T}(undef, Int.(d)), convert(NTuple{N,Int},d))

Base.copy(x::KnetArray) = KnetArray(copy(x.ptr))

# KnetArray(::KnetArray) creates a copy, convert returns an alias if possible
KnetArray(A::KnetArray{T,N})    where {T,N}   = KnetArray{T,N}(A)
KnetArray{T}(A::KnetArray{S,N}) where {T,N,S} = KnetArray{T,N}(A)
KnetArray{T,N}(x::KnetArray{T,N}) where {T,N} = copyto!(KnetArray{T}(undef,size(x)), x)
KnetArray{T,N}(x::KnetArray{S,N}) where {T,N,S} = copyto!(KnetArray{T}(undef,size(x)), x)

# KnetArray(::AbstractArray)
KnetArray(A::AbstractArray{T,N})    where {T,N}   = KnetArray{T,N}(A)
KnetArray{T}(A::AbstractArray{S,N}) where {T,N,S} = KnetArray{T,N}(A)
KnetArray{T,N}(x::AbstractArray{S,N}) where {T,N,S} = copyto!(KnetArray{T}(undef,size(x)), x)

# Array(::KnetArray)
import Base: Array
Array(A::KnetArray{T,N})    where {T,N}   = Array{T,N}(A)
Array{T}(A::KnetArray{S,N}) where {T,N,S} = Array{T,N}(A)
Array{T,N}(x::KnetArray{S,N}) where {T,N,S} = convert(Array{T,N}, copyto!(Array{S}(undef,size(x)), x))

gc(dev = gpu()) = CuArrays.reclaim(true)

# Conversions:
import Base: convert
# KnetArray <- KnetArray
convert(::Type{KnetArray}, x::KnetArray{T,N}) where {T,N} = x
convert(::Type{KnetArray{T}}, x::KnetArray{T,N}) where {T,N} = x
convert(::Type{KnetArray{T,N}}, x::KnetArray{T,N}) where {T,N} = x
convert(::Type{KnetArray{T}}, x::KnetArray{S,N}) where {T,N,S} = convert(KnetArray{T,N}, x)
convert(::Type{KnetArray{T,N}}, x::KnetArray{S,N}) where {T,N,S} = convert(KnetArray{T,N},convert(Array{S,N}, x))

# KnetArray <- AbstractArray
convert(::Type{KnetArray}, x::AbstractArray{T,N}) where {T,N} = convert(KnetArray{T,N}, x)
convert(::Type{KnetArray{T}}, x::AbstractArray{S,N}) where {T,N,S} = convert(KnetArray{T,N}, x)
convert(::Type{KnetArray{T,N}}, x::AbstractArray{S,N}) where {T,N,S} = copyto!(KnetArray{T,N}(undef,size(x)), convert(Array{T,N},x))

# Array <- KnetArray
convert(::Type{Array}, x::KnetArray{T,N}) where {T,N} = convert(Array{T,N}, x)
convert(::Type{Array{T}}, x::KnetArray{S,N}) where {T,N,S} = convert(Array{T,N}, x)
convert(::Type{Array{T,N}}, x::KnetArray{S,N}) where {T,N,S} = convert(Array{T,N},Array(x))

# Ptr <- KnetArray
import Base: cconvert, unsafe_convert, pointer
cconvert(::Type{Ptr{T}}, a::KnetArray) where {T} = a.ptr.buf
pointer(a::KnetArray{T}, i::Integer = 1) where {T} = convert(Ptr{T}, a.ptr.buf.ptr + i - 1)
unsafe_convert(::Type{Ptr{T}}, a::KnetArray{T}) where T = pointer(a)

# Reshape:
import Base: reshape, vec
reshape(a::KnetArray, dims::Dims) = KnetArray(reshape(a.ptr, dims))
reshape(a::KnetArray{<:Any,1}, dims::Tuple{Int64}) =
  length(a) == dims[1] ? a : error("Length must match in reshape")

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
hcat(a::NA, as::NA...)=cat(a,as...; dims=2)
hcat(a::NAK, as::NAK...)=throw(MethodError(hcat, (a, as...)))
vcat(a::NA, as::NA...)=cat(a,as...; dims=1)
vcat(a::NAK, as::NAK...)=throw(MethodError(vcat, (a, as...)))

# Ambiguity fix for abstractarray.jl:1066-1072
using Base: hvcat_fill, promote_typeof
vcat(X::Number, Xs::Number...) = hvcat_fill(Array{promote_typeof(X, Xs...)}(undef,1+length(Xs)), (X, Xs...))
hcat(X::Number, Xs::Number...) = hvcat_fill(Array{promote_typeof(X, Xs...)}(undef,1,1+length(Xs)), (X, Xs...))

function Base.copyto!(a::KnetArray, b::KnetArray)
  copyto!(a.ptr, b.ptr)
  return a
end

function Base.copyto!(a::KnetArray, b::AbstractArray)
  copyto!(a.ptr, b)
  return a
end

function Base.copyto!(a::AbstractArray, b::KnetArray)
  copyto!(a, b.ptr)
  return a
end


function Base.copyto!(x::KnetArray, a::Integer, y::KnetArray, b::Integer, c::Integer)
  copyto!(x.ptr, a, y.ptr, b, c)
  return x
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

knetwrap(x::CuArray) = KnetArray(x)
knetwrap(x) = x

knetunwrap(x::KnetArray) = x.ptr
knetunwrap(x) = x

import Base: getindex, setindex!

function getindex(A::KnetArray, I...) where {T}
  knetwrap(getindex(A.ptr, map(knetunwrap, I)...))
end

function setindex!(A::KnetArray, v, I...)
  knetwrap(setindex!(A.ptr, knetunwrap(v), map(knetunwrap, I)...))
end

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

c2i(d::Dims,i::AbstractArray{I}) where {I<:CartesianIndex} = Int32[(LinearIndices(d))[c.I...] for c in i]

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

# To stop fusing the following is needed.
# Primitives just need to override broadcasted for KnetArray types.
import .Broadcast: broadcasted
broadcasted(f, x::KnetArray) = throw(MethodError(broadcasted,(f,x)))
broadcasted(f, x::KnetArray, y::KnetArray) = throw(MethodError(broadcasted,(f,x,y)))
### These cause ambiguity with AutoGrad:
# broadcasted(f, x::KnetArray, y...) = throw(MethodError(broadcasted,(f,x,y...)))
# broadcasted(f, x::KnetArray, y) = throw(MethodError(broadcasted,(f,x,y)))
# broadcasted(f, x, y::KnetArray) = throw(MethodError(broadcasted,(f,x,y)))

import .Broadcast: copyto!, broadcasted
using .Broadcast: Broadcasted
# This fixes (x .- log.(sum(exp.(x),dims=dims)))
broadcasted(f, x::KnetArray, y::Broadcasted) = broadcasted(f, x, copy(y))
broadcasted(f, x::Broadcasted, y::KnetArray) = broadcasted(f, copy(x), y)

# # This fixes ambiguity with AutoGrad
# # But then creates more ambiguity as given next
# using AutoGrad: broadcast_r
# broadcasted(f, x::Value, y::KnetArray) = broadcast_r(f,x,y)
# broadcasted(f, x::KnetArray, y::Value) = broadcast_r(f,x,y)
# # This fixes (dy.*((2 .* y) .* x .- convert(eltype(x), 2 / √π)))
# # TODO: Do we have to do this for each f?
# broadcasted(f::typeof(*), x::KnetArray, y::Value) = broadcast_r(f,x,y)
# broadcasted(f::typeof(*), x::Value, y::KnetArray) = broadcast_r(f,x,y)

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
show(io::IO, a::KnetArray) = show(io, KnetDisplay(a))
show(io::IO, m::MIME"text/plain", a::KnetArray) = show(io, m, KnetDisplay(a))
summary(io::IO, x::Value{A}) where {A<:KnetArray} = print(io, Base.dims2string(size(x)), " ", typeof(x))
