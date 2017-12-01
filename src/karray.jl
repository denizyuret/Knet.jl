"""

    KnetArray{T}(dims)
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

* Array operations: ==, !=, cat, convert, copy, copy!, deepcopy,
  display, eachindex, eltype, endof, fill!, first, hcat, isapprox,
  isempty, length, ndims, ones, pointer, rand!, randn!, reshape,
  similar, size, stride, strides, summary, vcat, vec, zeros.
  (cat(i,x,y) supported for i=1,2.)

* Math operators: (-), abs, abs2, acos, acosh, asin, asinh, atan,
  atanh, cbrt, ceil, cos, cosh, cospi, erf, erfc, erfcinv, erfcx,
  erfinv, exp, exp10, exp2, expm1, floor, log, log10, log1p, log2,
  round, sign, sin, sinh, sinpi, sqrt, tan, tanh, trunc

* Broadcasting operators: (.*), (.+), (.-), (./), (.<), (.<=), (.!=),
  (.==), (.>), (.>=), (.^), max, min.  (Boolean operators generate
  outputs with same type as inputs; no support for KnetArray{Bool}.)

* Reduction operators: countnz, maximum, mean, minimum, prod, sum,
  sumabs, sumabs2, vecnorm.
    
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
type KnetArray{T,N}
    ptr::KnetPtr
    dims::NTuple{N,Int}
end

# Note: I removed <: AbstractArray{T,N} after I painfully discovered
# some inefficient AbstractArray methods inherited unintentionally.
# It is better to define a few extra methods to keep a tighter control
# on what methods exactly get called for KnetArrays.


# Aliases:

@typealias6 KnetMatrix{T} KnetArray{T,2}
@typealias6 KnetVector{T} KnetArray{T,1}
@typealias6 KnetVecOrMat{T} Union{KnetVector{T}, KnetMatrix{T}}

# Constructors:
import Base: convert
# Internal constructor defines KnetArray{T,N}(ptr,dims)
# These define KnetArray{T,N}(dims) and KnetArray{T,N}(d...)
if VERSION >= v"0.5.0-dev+7720"; @eval begin
    (::Type{KnetArray{T,N}}){T,N}(d::NTuple{N,Int})=KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), d)
    (::Type{KnetArray{T,N}}){T,N}(d::Int...) = KnetArray{T,N}(d)
    (::Type{KnetArray{T,N}}){T,N}(d::Integer...) = KnetArray{T,N}(convert(Tuple{Vararg{Int}}, d))
end; else; @eval begin
    convert{T,N}(::Type{KnetArray{T,N}},d::NTuple{N,Int})=KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), d)
    convert{T,N}(::Type{KnetArray{T,N}},d::Int...)=convert(KnetArray{T,N},d)
    convert{T,N}(::Type{KnetArray{T,N}},d::Integer...)=convert(KnetArray{T,N},convert(Tuple{Vararg{Int}}, d))
end; end
# These define KnetArray{T}(dims) and KnetArray{T}(d...)
if VERSION >= v"0.5.0-dev+7720"; @eval begin
    (::Type{KnetArray{T}}){T,N}(d::NTuple{N,Int}) = KnetArray{T,N}(d)
    (::Type{KnetArray{T}}){T}(d::Int...) = KnetArray{T}(d)
    (::Type{KnetArray{T}}){T}(d::Integer...) = KnetArray{T}(convert(Tuple{Vararg{Int}}, d))
end; else; @eval begin
    convert{T,N}(::Type{KnetArray{T}},d::NTuple{N,Int})=KnetArray{T,N}(KnetPtr(sizeof(T)*prod(d)), d)
    convert{T}(::Type{KnetArray{T}},d::Int...)=KnetArray{T}(d)
    convert{T}(::Type{KnetArray{T}},d::Integer...)=KnetArray{T}(convert(Tuple{Vararg{Int}}, d))
end; end
# These define KnetArray(T,dims) and KnetArray(T,d...)
KnetArray{T,N}(::Type{T}, d::NTuple{N,Int}) = KnetArray{T,N}(d)
KnetArray{T}(::Type{T}, d::Int...)=KnetArray(T,d)
KnetArray{T}(::Type{T}, d::Integer...)=KnetArray(T,convert(Tuple{Vararg{Int}},d))

# Conversions:
import Base: convert, reshape, vec, unsafe_convert, pointer
# KnetArray <- KnetArray
convert{T,N}(::Type{KnetArray}, x::KnetArray{T,N}) = x
convert{T,N}(::Type{KnetArray{T}}, x::KnetArray{T,N}) = x
convert{T,N}(::Type{KnetArray{T,N}}, x::KnetArray{T,N}) = x
convert{T,N,S}(::Type{KnetArray{T}}, x::KnetArray{S,N}) = convert(KnetArray{T,N}, x)
convert{T,N,S}(::Type{KnetArray{T,N}}, x::KnetArray{S,N}) = convert(KnetArray{T,N},unsafe_copy!(Array{S}(size(x)), 1, x, 1, length(x)))

reshape{T}(a::KnetArray{T}, dims::Dims)
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
# KnetArray <- AbstractArray
convert{T,N}(::Type{KnetArray}, x::AbstractArray{T,N}) = convert(KnetArray{T,N}, x)
convert{T,N,S}(::Type{KnetArray{T}}, x::AbstractArray{S,N}) = convert(KnetArray{T,N}, x)
convert{T,N,S}(::Type{KnetArray{T,N}}, x::AbstractArray{S,N}) = unsafe_copy!(KnetArray{T}(size(x)), 1, convert(Array{T,N},x), 1, length(x))
# Array <- KnetArray
convert{T,N}(::Type{Array}, x::KnetArray{T,N}) = convert(Array{T,N}, x)
convert{T,N,S}(::Type{Array{T}}, x::KnetArray{S,N}) = convert(Array{T,N}, x)
convert{T,N,S}(::Type{Array{T,N}}, x::KnetArray{S,N}) = convert(Array{T,N},unsafe_copy!(Array{S}(size(x)), 1, x, 1, length(x)))
# Ptr <- KnetArray
unsafe_convert{T}(::Type{Ptr{T}}, a::KnetArray) = unsafe_convert(Ptr{T}, pointer(a))
pointer{T}(a::KnetArray{T})=convert(Ptr{T}, a.ptr.ptr)
pointer{T}(a::KnetArray{T},i)=convert(Ptr{T}, a.ptr.ptr + (i-1)*sizeof(T))

# AbstractArray interface
import Base: eachindex, eltype, endof, fill!, first, isempty, length, ndims, ones, similar, size, stride, strides, zeros, (==), isapprox #, linearindexing
eachindex(a::KnetArray) = (1:length(a))
eltype{T}(::KnetArray{T})=T
eltype{T}(::Type{KnetArray{T}}) = T
eltype{T,n}(::Type{KnetArray{T,n}}) = T
endof(a::KnetArray) = length(a)
fill!{T}(a::KnetArray{T},x)=(a[:]=T(x);a)
first(a::KnetArray) = a[1]
# AutoGrad leaves `first` as a compound proc calling start which doesn't work with KnetArrays
@primitive  first(x::KnetArray),dy,y  AutoGrad.ungetindex(x,dy,1)
isempty(a::KnetArray) = (0==length(a))
length(a::KnetArray)=prod(size(a))
# linearindexing(::KnetArray)=Base.LinearFast() # deprecated in Julia6
ndims{T,N}(a::KnetArray{T,N})=N
ones{T}(a::KnetArray{T})=fill!(similar(a),one(T))
similar(a::KnetArray, T, dims::Dims)      = KnetArray(T, dims)
similar(a::KnetArray, T, dims::Int...)    = similar(a, T, dims)
similar(a::KnetArray, T)                  = similar(a, T, size(a))
similar{T}(a::KnetArray{T})               = similar(a, T, size(a))
similar{T}(a::KnetArray{T}, dims::Dims)   = similar(a, T, dims)
similar{T}(a::KnetArray{T}, dims::Int...) = similar(a, T, dims)
size(a::KnetArray)=a.dims
size{T,N}(a::KnetArray{T,N},i::Integer)=(if i>N; 1; else; size(a)[i]; end)
stride{T,N}(a::KnetArray{T,N},i::Integer)=(if i>N; length(a); else; s=1; for n=1:(i-1); s*=size(a,n); end; s; end)
strides{T,N}(a::KnetArray{T,N})=ntuple(n->stride(a,n), N)
zeros{T}(a::KnetArray{T})=fill!(similar(a),zero(T))

# Comparisons
(==){T}(a::KnetArray{T},b::KnetArray{T})=(size(a)==size(b) && vecnorm(a-b)==0)
(==)(a::AbstractArray,b::KnetArray)=(size(a)==size(b) && a==Array(b))
(==)(a::KnetArray,b::AbstractArray)=(size(a)==size(b) && Array(a)==b)
# Adapted from base/linalg/generic.jl:589
isapprox{T}(a::KnetArray{T}, b::KnetArray{T}; rtol=sqrt(eps(T)), atol=T(0))=(size(a)==size(b) && vecnorm(a-b) <= atol + rtol * max(vecnorm(a), vecnorm(b)))
isapprox(a::AbstractArray,b::KnetArray;o...)=(size(a)==size(b) && isapprox(a,Array(b);o...))
isapprox(a::KnetArray,b::AbstractArray;o...)=(size(a)==size(b) && isapprox(Array(a),b;o...))



# Concatenation:
import Base: hcat, vcat, cat

# Need to extend cat definitions from AutoGrad/src/base/abstractarray.jl:
const NARK = Union{Number,AbstractArray,Rec,KnetArray}
cat(::Type{Grad{1}},a::KnetArray,as::KnetArray...)=nothing # ambiguity fix
cat(::Type{Grad{1}},a::NARK...)=nothing # ambiguity fix
cat{N}(::Type{Grad{N}},y1::NARK,y::NARK,dims::NARK,x::NARK...)=AutoGrad.uncat(y1,N-1,dims,x...) # ambiguity fix
cat(dims, X::NARK...)=AutoGrad.cat_r(dims, X...)

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
function hcat{T}(A::KnetVecOrMat{T}...)
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
        copy!(B, pos, Ak, 1, n)
        pos += n
    end
    return B
end

function vcat{T}(A::KnetVector{T}...)
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
        copy!(B, pos, Ak, 1, n)
        pos += n
    end
    return B
end

function vcat{T}(A::KnetVecOrMat{T}...)
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

cat{T}(d::Type{Grad{1}}, a1::KnetVecOrMat{T}, a::KnetVecOrMat{T}...)=nothing # ambiguity fix

function cat{T}(d, a1::KnetVecOrMat{T}, a::KnetVecOrMat{T}...)
    if     d==1 || d==Val{1}; vcat(a1, a...)
    elseif d==2 || d==Val{2}; hcat(a1, a...)
    else error("cat($d,a...) not implemented.")
    end
end

# Avoid using Base for unimplemented cat methods:

using AutoGrad: NA # Union{Number,AbstractArray}
const NAK = Union{Number,AbstractArray,KnetArray}
# cat(d, a::NA, as::NA...)=Base.cat_t(d, prom_(a...), a...) # defined in AutoGrad
cat(d, a::NAK, as::NAK...)=throw(MethodError(cat, (a, as...)))
hcat(a::NA, as::NA...)=cat(2,a,as...)
hcat(a::NAK, as::NAK...)=throw(MethodError(hcat, (a, as...)))
vcat(a::NA, as::NA...)=cat(1,a,as...)
vcat(a::NAK, as::NAK...)=throw(MethodError(vcat, (a, as...)))

# Utilities:

# Generalizing low level copy using linear indexing to/from gpu arrays:
# copy!{T}(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer)
# Note that this is an unsafe operation, no argument or bounds checking performed.
# Defined in Base:
# unsafe_copy!{T}(dest::Ptr{T}, src::Ptr{T}, n) at array.jl:73
# unsafe_copy!{T}(dest::Array{T,N}, doffs, src::Array{T,N}, soffs, n) at array.jl:79

import Base: unsafe_copy!, copy, copy!
@typealias6 KorA{T} Union{KnetArray{T},Array{T}}

function copy!{T}(dest::KorA{T}, doffs::Integer, src::KorA{T}, soffs::Integer, n::Integer)
    if n == 0; return dest; end
    if n < 0; throw(ArgumentError()); end
    if soffs < 1 || doffs < 1 || soffs+n-1 > length(src) || doffs+n-1 > length(dest)
        throw(BoundsError())
    end
    unsafe_copy!(dest, doffs, src, soffs, n)
end

function copy!{T}(dest::KorA{T}, src::KorA{T})
    if length(dest) < length(src); throw(BoundsError()); end
    copy!(dest, 1, src, 1, length(src))
end

function copy(a::KnetArray)
    unsafe_copy!(similar(a),1,a,1,length(a))
end

# unsafe_copy! does no bounds checking, the callers must.
function unsafe_copy!{T}(dest::KnetArray{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int)
    @cuda(cudart,cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
          pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), 1)
    return dest
end
function unsafe_copy!{T}(dest::Array{T}, doffs::Int, src::KnetArray{T}, soffs::Int, n::Int)
    @cuda(cudart,cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
          pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), 2)
    return dest
end
function unsafe_copy!{T}(dest::KnetArray{T}, doffs::Int, src::KnetArray{T}, soffs::Int, n::Int)
    @cuda(cudart,cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
          pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), 3)
    return dest
end

# This will make deepcopy work properly
Base.deepcopy_internal(x::KnetArray, s::ObjectIdDict)=if haskey(s,x); s[x]; else; copy(x); end

function cudadir(a,b)
    deva = isa(a,KnetArray) && a.ptr.dev >= 0
    devb = isa(b,KnetArray) && b.ptr.dev >= 0
    if !deva && !devb; return 0
    elseif deva && !devb; return 1
    elseif !deva && devb; return 2
    elseif deva && devb;  return 3
    end
end


# Hack for printing without copying the whole KnetArray and without inheriting AbstractArray:
import Base: display, summary, getindex, size
type KnetDisplay{T,N} <: AbstractArray{T,N}; a::KnetArray{T,N}; end
getindex(a::KnetDisplay, i...) = getindex(a.a, i...)
size(a::KnetDisplay) = size(a.a)
summary(a::KnetDisplay) = summary(a.a)
summary(a::KnetArray) = string(Base.dims2string(size(a)), " ", typeof(a))
display(a::KnetArray) = display(KnetDisplay(a))

# Hack for JLD file load/save of KnetArrays:
if Pkg.installed("JLD") != nothing
    import JLD: writeas, readas
    type KnetJLD; a::Array; end
    writeas(c::KnetArray) = KnetJLD(Array(c))
    readas(d::KnetJLD) = (gpu() >= 0 ? KnetArray(d.a) : d.a)
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
# @primitive convert{A<:AbstractArray,K<:KnetArray}(T::Type{K}, x::Rec{A}),dy 0 Array(dy)
# @primitive convert{A<:AbstractArray,K<:KnetArray}(T::Type{A}, x::Rec{K}),dy 0 KnetArray(dy)

# So we will define gradients for convert, KnetArray, Array manually:
Base.Array{K<:KnetArray}(x::Rec{K})=convert(Array,x)
KnetArray{A<:AbstractArray}(x::Rec{A})=convert(KnetArray,x)
let convert_r = recorder(convert)
    global convert
    convert(::Type{Grad{2}},dy,y,T,x) = convert(typeof(AutoGrad.getval(x)),dy)
    # This does not work, it breaks the Node(::Rec) constructor, so we define Knet specific version.
    # convert(T::Type, x::Rec) = convert_r(T,x)
    convert{A<:AbstractArray,K<:KnetArray}(::Type{A},x::Rec{K})=convert_r(A,x)
    convert{A<:AbstractArray,K<:KnetArray}(::Type{K},x::Rec{A})=convert_r(K,x)
end


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

function getindex{T}(A::KnetArray{T}, I::Real)
    J = Int(I)
    if !(1 <= J <= length(A)); throw(BoundsError(A,J)); end
    unsafe_copy!(T[0], 1, A, J, 1)[1]
end

function setindex!{T}(A::KnetArray{T}, v, I::Real)
    J = Int(I)
    if !(1 <= J <= length(A)); throw(BoundsError(A,J)); end
    unsafe_copy!(A, J, T[v], 1, 1)
end

## Indexing with Tuple{Real}
# Julia #14770
# If I is shorter than ndims(A) but longer than 1 the remaining indices assumed =1
# Also extra 1's at the end of I are ignored

function getindex{T}(A::KnetArray{T}, I::Real...)
    J = ntuple(i->Int(I[i]), length(I)) # deprecated: Base.to_indexes(I...)
    @inbounds for j=1:length(J)
        if !(1 <= J[j] <= size(A,j)); throw(BoundsError(A,J)); end
    end
    i = sub2ind(size(A), J...)
    unsafe_copy!(T[0], 1, A, i, 1)[1]
end

function setindex!{T}(A::KnetArray{T}, v, I::Real...)
    J = ntuple(i->Int(I[i]), length(I)) # deprecated: Base.to_indexes(I...)
    @inbounds for j=1:length(J)
        if !(1 <= J[j] <= size(A,j)); throw(BoundsError(A,J)); end
    end
    i = sub2ind(size(A), J...)
    unsafe_copy!(A, i, T[v], 1, 1)
end

## Indexing with CartesianIndex: calls Tuple{Real}

function getindex{T}(A::KnetArray{T}, c::CartesianIndex)
    getindex(A, c.I...)
end

function setindex!{T}(A::KnetArray{T}, v, c::CartesianIndex)
    setindex!(A, v, c.I...)
end

## Indexing with AbstractUnitRange
# We will implement indexing ranges as views not copies, if possible (when contiguous).
# For contiguous memory without stride all but the last >1 dimension must be full
# The original getindex(a,i:j...) for AbstractArray copies:
# function _getindex(l::LinearIndexing, A::AbstractArray, I::Union{Real, AbstractArray, Colon}...)
# in abstractarray.jl:487,multidimensional.jl:184.

if VERSION < v"0.5.0"
    @typealias6 AbstractUnitRange UnitRange
end

function getindex{T}(A::KnetArray{T}, I::AbstractUnitRange)
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

function setindex!{T}(A::KnetArray{T}, v::Real, I::AbstractUnitRange)
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    if length(I)==0; return A; end
    unsafe_setindex!(A,T(v),I)
end

function setindex!{T}(A::KnetArray{T}, v::Real, I::AbstractUnitRange{Bool}) # julia4 ambig fix
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    if length(I)==0; return A; end
    unsafe_setindex!(A,T(v),I)
end

function setindex!{T}(A::KnetArray{T}, v, I::AbstractUnitRange)
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    if length(v)!=length(I); throw(DimensionMismatch()); end
    if length(I)==0; return A; end
    if eltype(v)!=T; v = convert(Array{T},v); end
    unsafe_copy!(A,first(I),v,1,length(I))
end

## Indexing with Colon
# Note that getindex(a,:) returns a view not a copy

function getindex(A::KnetArray, I::Colon)
    reshape(A,length(A))
end

function setindex!{T}(A::KnetArray{T}, v::Real, I::Colon)
    if length(A)==0; return A; end
    unsafe_setindex!(A, T(v), 1:length(A))
end

function setindex!{T}(A::KnetArray{T}, v, I::Colon)
    if length(v)!=length(A); throw(DimensionMismatch()); end
    if length(v)==0; return A; end
    if eltype(v)!=T; v = convert(Array{T},v); end
    unsafe_copy!(A,1,v,1,length(A))
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

function checkbetween{I<:Integer,L<:Integer,H<:Integer}(i::AbstractArray{I},lo::L,hi::H)
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

function getindex{T,I<:Real}(x::KnetArray{T}, i::AbstractArray{I})
    y = similar(x, size(i))
    if isempty(y); return y; end
    i = Array{Int32}(i)
    checkbetween(i, 1, length(x))
    unsafe_getindex!(x,y,KnetArray{Int32}(i))
    return y
end

function setindex!{T,I<:Real}(x::KnetArray{T}, y::Real, i::AbstractArray{I})
    if isempty(i); return x; end
    i = Array{Int32}(i)
    checkbetween(i, 1, length(x))
    unsafe_setindex!(x,T(y),KnetArray{Int32}(i))
    return x
end

function setindex!{T,I<:Real}(x::KnetArray{T}, y, i::AbstractArray{I})
    if length(y) != length(i); throw(DimensionMismatch()); end
    if isempty(i); return x; end
    i = Array{Int32}(i)
    checkbetween(i, 1, length(x))
    unsafe_setindex!(x,KnetArray{T}(y),KnetArray{Int32}(i))
    return x
end

## Indexing with (Colon,AbstractVector{Real}) calls (Colon,KnetArray{Int32}) after bound checking

function getindex{T,I<:Real}(x::KnetMatrix{T}, c::Colon, i::AbstractVector{I})
    xrows,xcols = size(x); ycols = length(i)
    y = similar(x, xrows, ycols)
    if isempty(y); return y; end
    i = Array{Int32}(i)
    checkbetween(i,1,xcols)
    unsafe_getindex!(x,y,c,KnetArray{Int32}(i))
    return y
end

function setindex!{T,I<:Real}(x::KnetMatrix{T}, y::Real, c::Colon, i::AbstractVector{I})
    if isempty(i); return x; end
    xrows,xcols = size(x); ycols=length(i)
    i = Array{Int32}(i)
    checkbetween(i,1,xcols)
    unsafe_setindex!(x,T(y),c,KnetArray{Int32}(i))
    return x
end

function setindex!{T,I<:Real}(x::KnetMatrix{T}, y, c::Colon, i::AbstractVector{I})
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

## Indexing with (AbstractVector{Real},Colon) calls (KnetArray{Int32},Colon) after bound checking

function getindex{T,I<:Real}(x::KnetMatrix{T}, i::AbstractVector{I}, c::Colon)
    xrows,xcols = size(x); yrows = length(i)
    y = similar(x, yrows, xcols)
    if isempty(y); return y; end
    i = Array{Int32}(i)
    checkbetween(i,1,xrows)
    unsafe_getindex!(x,y,KnetArray{Int32}(i),c)
    return y
end

function setindex!{T,I<:Real}(x::KnetMatrix{T}, y::Real, i::AbstractVector{I}, c::Colon)
    if isempty(i); return x; end
    xrows,xcols = size(x); yrows=length(i)
    i = Array{Int32}(i)
    checkbetween(i,1,xrows)
    unsafe_setindex!(x,T(y),KnetArray{Int32}(i),c)
    return x
end

function setindex!{T,I<:Real}(x::KnetMatrix{T}, y, i::AbstractVector{I}, c::Colon)
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

## Indexing with AbstractArray{CartesianIndex} calls AbstractArray{Real}

c2i{I<:CartesianIndex}(d::Dims,i::AbstractArray{I})=Int32[sub2ind(d,c.I...) for c in i]
getindex{T,I<:CartesianIndex}(x::KnetArray{T}, i::AbstractArray{I})=getindex(x, c2i(size(x),i))
setindex!{T,I<:CartesianIndex}(x::KnetArray{T}, y, i::AbstractArray{I})=setindex!(x, y, c2i(size(x),i))

## Indexing with Empty Array or other unrecognized AbstractArray calls AbstractArray{Real}

getindex{T}(x::KnetArray{T}, i::AbstractArray)=getindex(x, Array{Int32}(i))
setindex!{T}(x::KnetArray{T}, y, i::AbstractArray)=setindex!(x, y, Array{Int32}(i))

## Indexing with StepRange calls AbstractArray{Real}

function getindex{T}(A::KnetArray{T}, I::StepRange)
    getindex(A, collect(I))
end

function setindex!{T,R<:Real}(A::KnetArray{T}, v::Real, I::StepRange{R}) # julia4 ambiguity fix
    setindex!(A, v, collect(I))
end

function setindex!{T}(A::KnetArray{T}, v::Real, I::StepRange{Bool}) # julia4 ambiguity fix
    setindex!(A, v, collect(I))
end

function setindex!{T}(A::KnetArray{T}, v, I::StepRange)
    setindex!(A, v, collect(I))
end

## Indexing with (StepRange,Colon) calls (AbstractArray{Real},Colon)

function getindex{T}(A::KnetMatrix{T}, I::StepRange, c::Colon)
    getindex(A, collect(I), c)
end

function setindex!{T,R<:Real}(A::KnetMatrix{T}, v::Real, I::StepRange{R}, c::Colon) # julia4 ambig fix
    setindex!(A, v, collect(I), c)
end

function setindex!{T}(A::KnetMatrix{T}, v::Real, I::StepRange{Bool}, c::Colon) # julia4 ambig fix
    setindex!(A, v, collect(I), c)
end

function setindex!{T}(A::KnetMatrix{T}, v, I::StepRange, c::Colon)
    setindex!(A, v, collect(I), c)
end

## Indexing with (Colon,StepRange) calls (Colon,AbstractArray{Real})

function getindex{T}(A::KnetMatrix{T}, c::Colon, I::StepRange)
    getindex(A, c, collect(I))
end

function setindex!{T,R<:Real}(A::KnetMatrix{T}, v::Real, c::Colon, I::StepRange{R}) # julia4 ambig fix
    setindex!(A, v, c, collect(I))
end

function setindex!{T}(A::KnetMatrix{T}, v::Real, c::Colon, I::StepRange{Bool}) # julia4 ambig fix
    setindex!(A, v, c, collect(I))
end

function setindex!{T}(A::KnetMatrix{T}, v, c::Colon, I::StepRange)
    setindex!(A, v, c, collect(I))
end

## Indexing with AbstractArray{Bool} calls KnetArray{Int32}; no need for bound checking

function getindex{T}(x::KnetArray{T}, i::AbstractArray{Bool})
    if length(i) != length(x); throw(BoundsError(x,i)); end
    j = find(i)
    y = similar(x, length(j))
    if !isempty(y); unsafe_getindex!(x,y,KnetArray{Int32}(j)); end
    return y
end

function setindex!{T}(x::KnetArray{T}, y::Real, i::AbstractArray{Bool})
    if length(i) != length(x); throw(DimensionMismatch()); end
    j = find(i)
    if !isempty(j); unsafe_setindex!(x,T(y),KnetArray{Int32}(j)); end
    return x
end

function setindex!{T}(x::KnetArray{T}, y, i::AbstractArray{Bool})
    if length(i) != length(x); throw(DimensionMismatch()); end
    j = find(i)
    if length(j) != length(y); throw(BoundsError(y,j)); end
    if !isempty(j); unsafe_setindex!(x,KnetArray{T}(y),KnetArray{Int32}(j)); end
    return x
end

## Indexing with (Colon,AbstractVector{Bool}) calls (Colon,KnetArray{Int32}); no need for bound checking

function getindex{T}(x::KnetMatrix{T}, c::Colon, i::AbstractVector{Bool})
    xrows,xcols = size(x)
    if length(i) != xcols; throw(BoundsError(x,(:,i))); end
    j = find(i); ycols = length(j)
    y = similar(x, xrows, ycols)
    if !isempty(y); unsafe_getindex!(x,y,c,KnetArray{Int32}(j)); end
    return y
end

function setindex!{T}(x::KnetMatrix{T}, y::Real, c::Colon, i::AbstractVector{Bool})
    xrows,xcols = size(x)
    if length(i) != xcols; throw(BoundsError(x,(:,i))); end
    j = find(i)
    if !isempty(j); unsafe_setindex!(x,T(y),c,KnetArray{Int32}(j)); end
    return x
end

function setindex!{T}(x::KnetMatrix{T}, y, c::Colon, i::AbstractVector{Bool})
    if ndims(y) != 2; throw(DimensionMismatch()); end
    xrows,xcols = size(x); yrows,ycols=size(y)
    if yrows != xrows; throw(DimensionMismatch()); end
    if length(i) != xcols; throw(BoundsError(x,(:,i))); end
    j = find(i)
    if ycols != length(j); throw(DimensionMismatch()); end
    if !isempty(y); unsafe_setindex!(x,KnetArray{T}(y),c,KnetArray{Int32}(j)); end
    return x
end

## Indexing with (AbstractVector{Bool},Colon) calls (KnetArray{Int32},Colon); no need for bound checking

function getindex{T}(x::KnetMatrix{T}, i::AbstractVector{Bool}, c::Colon)
    xrows,xcols = size(x)
    if length(i) != xrows; throw(BoundsError(x,(i,:))); end
    j = find(i); yrows = length(j)
    y = similar(x, yrows, xcols)
    if !isempty(y); unsafe_getindex!(x,y,KnetArray{Int32}(j),c); end
    return y
end

function setindex!{T}(x::KnetMatrix{T}, y::Real, i::AbstractVector{Bool}, c::Colon)
    xrows,xcols = size(x)
    if length(i) != xrows; throw(BoundsError(x,(i,:))); end
    j = find(i)
    if !isempty(j); unsafe_setindex!(x,T(y),KnetArray{Int32}(j),c); end
    return x
end

function setindex!{T}(x::KnetMatrix{T}, y, i::AbstractVector{Bool}, c::Colon)
    if ndims(y) != 2; throw(DimensionMismatch()); end
    xrows,xcols = size(x); yrows,ycols=size(y)
    if ycols != xcols; throw(DimensionMismatch()); end
    if length(i) != xrows; throw(BoundsError(x,(i,:))); end
    j = find(i)
    if yrows != length(j); throw(DimensionMismatch()); end
    if !isempty(y); unsafe_setindex!(x,KnetArray{T}(y),KnetArray{Int32}(j),c); end
    return x
end

## Indexing with KnetArray{T} for logicals calls KnetArray{Int32}
# Need this because (k.<0) returns KnetArray{T} instead of BitArray

function getindex{T}(x::KnetArray{T}, i::KnetArray{T})
    if length(i) != length(x); throw(BoundsError(x,i)); end
    j = find(Array(i))
    y = similar(x, length(j))
    if !isempty(y); unsafe_getindex!(x,y,KnetArray{Int32}(j)); end
    return y
end

function setindex!{T}(x::KnetArray{T}, y::Real, i::KnetArray{T})
    if length(i) != length(x); throw(DimensionMismatch()); end
    j = find(Array(i))
    if !isempty(j); unsafe_setindex!(x,T(y),KnetArray{Int32}(j)); end
    return x
end

function setindex!{T}(x::KnetArray{T}, y, i::KnetArray{T})
    if length(i) != length(x); throw(DimensionMismatch()); end
    j = find(Array(i))
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
setindex!{T<:Real}(A::KnetMatrix, B::Real, I1::Colon, I2::AbstractUnitRange{T})=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B, I1::Colon, I2::AbstractUnitRange)=setindex2!(A,B,I1,I2)
setindex!(A::KnetMatrix, B::Number, I1::Colon, I2::AbstractUnitRange)=setindex2!(A,B,I1,I2)
getindex(A::KnetMatrix, I1::AbstractUnitRange, I2::Colon)=getindex2(A,I1,I2)
setindex!(A::KnetMatrix, B::Real, I1::AbstractUnitRange{Bool}, I2::Colon)=setindex2!(A,B,I1,I2)
setindex!{R<:Real}(A::KnetMatrix, B::Real, I1::AbstractUnitRange{R}, I2::Colon)=setindex2!(A,B,I1,I2)
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

function getindex2{T}(A::KnetMatrix{T}, I1::Index3, I2::Index3)
    (nelts,nrows,ncols,firstindex,astep) = indexparams(A,I1,I2)
    B1 = isa(I1,Colon) ? size(A,1) : length(I1)
    B2 = isa(I2,Colon) ? size(A,2) : length(I2)
    Bsize = isa(I1,Real) ? (B2,) : isa(I2,Real) ? (B1,) : (B1,B2)
    if VERSION < v"0.5.0" && isa(I1,Real)
        Bsize = (B1,B2)
    end
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

function setindex2!{T}(A::KnetMatrix{T}, B, I1::Index3, I2::Index3)
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
                @cuda(cudart,cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
                      aptr0, B, nelts*sizeof(T), cudadir(A,B))
            end
        elseif nrows > 0 && ncols > 0
            nrows *= sizeof(T); astep *= sizeof(T)
            @knet8(xcopy,(Cint,Cint,Cptr,Cint,Cptr,Cint), nrows, ncols, B, nrows, aptr0, astep)
        end
    end
    return A
end

function indexparams{T,N}(A::KnetArray{T,N}, I::Index3...)
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
    firstindex = sub2ind(size(A),subs1...)
    return (nelts,nrows,ncols,firstindex,astep)
end


# These two are not sufficient in spite of what the documentation says:
# display goes into an infinite loop!
# getindex{T}(A::KnetArray{T}, i::Int)=unsafe_copy!(T[0], 1, A, i, 1)[1]
# setindex!{T}(A::KnetArray{T}, v, i::Int)=unsafe_copy!(A, i, T[v], 1, 1)


# AutoGrad functions:
import AutoGrad: zeroslike, sum_outgrads, UngetIndex, unary_nd, indexed_function, isequivalent, _dbg, ssize
zeroslike(a::KnetArray)=zeros(a)
unary_nd(f, x::KnetArray, eps) = reshape(eltype(x)[unary_nd(indexed_function(f, x, i), x[i], eps) for i in 1:length(x)], size(x))
isequivalent(x::Union{KnetArray,AbstractArray}, y::Union{KnetArray,AbstractArray}; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))
_dbg(a::KnetArray) = "K"*_dbg(Array(a))

# Note that KnetArray sum_outgrads is overwriting, i.e. does not support higher order gradients.
sum_outgrads{T}(a::KnetArray{T},b::KnetArray{T})=axpy!(1,b,a) # (a+b)

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

sum_outgrads_karray{T<:CartesianIndex}(A::KnetArray, X, I::AbstractArray{T})=sum_outgrads_karray(A,X,c2i(size(A),I))

for F in (32,64); T=Symbol("Float$F"); @eval begin

    function sum_outgrads_karray{R<:Real}(A::KnetArray{$T}, X, I::AbstractArray{R})
        I = KnetArray{Int32}(I)
        X = KnetArray{$T}(X)
        @knet8($("addents_$F"),(Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               length(I), I, A, X)
        return A
    end

    function sum_outgrads_karray{R<:Real}(A::KnetArray{$T}, X, ::Colon, I::AbstractArray{R})
        I = KnetArray{Int32}(I)
        X = KnetArray{$T}(X)
        @knet8($("addcols_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(A,1), size(A,2), length(I), I, A, X)
        return A
    end

    function sum_outgrads_karray{R<:Real}(A::KnetArray{$T}, X, I::AbstractArray{R}, ::Colon)
        I = KnetArray{Int32}(I)
        X = KnetArray{$T}(X)
        @knet8($("addrows_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(A,1), size(A,2), length(I), I, A, X)
        return A
    end

    sum_outgrads_karray(A::KnetArray{$T}, X, I::AbstractArray{Bool})=sum_outgrads_karray(A,X,find(I))
    sum_outgrads_karray(A::KnetArray{$T}, X, c::Colon, I::AbstractArray{Bool})=sum_outgrads_karray(A,X,c,find(I))
    sum_outgrads_karray(A::KnetArray{$T}, X, I::AbstractArray{Bool}, c::Colon)=sum_outgrads_karray(A,X,find(I),c)

end; end

# To prevent RSI
ka = KnetArray
export ka
