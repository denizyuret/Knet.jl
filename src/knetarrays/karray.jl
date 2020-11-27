export KnetArray, KnetMatrix, KnetVector, KnetVecOrMat, DevArray, ka
import CUDA: CUDA, CuArray, device
import Base: Array, convert, unsafe_convert, pointer
using CUDA: cuMemcpyHtoD_v2, cuMemcpyDtoH_v2, cuMemcpyDtoD_v2, CuPtr
using Base: unsafe_wrap
# include("kptr.jl") ## KnetPtr

"""

    KnetArray{T}(undef,dims)
    KnetArray(a::AbstractArray)
    Array(k::KnetArray)

Container for GPU arrays that supports most of the AbstractArray interface.  The constructor
allocates a KnetArray in the currently active device, as specified by `CUDA.device()`.
KnetArrays and Arrays can be converted to each other as shown above, which involves copying
to and from the GPU memory.  Only Float32/64 KnetArrays are fully supported.

KnetArrays use the CUDA.jl package for allocation and some operations. Currently some of
the custom CUDA kernels that implement elementwise, broadcasting, and reduction operations
for KnetArrays work faster. Once these are improved in CUDA.jl, KnetArrays will be retired.

# Supported functions:

* Indexing: getindex, setindex! with the following index types:
  * 1-D: Real, Colon, OrdinalRange, AbstractArray{Real}, AbstractArray{Bool}, CartesianIndex, AbstractArray{CartesianIndex}, EmptyArray, KnetArray{Int32} (low level), KnetArray{0/1} (using float for BitArray) (1-D includes linear indexing of multidimensional arrays)
  * 2-D: (Colon,Union{Real,Colon,OrdinalRange,AbstractVector{Real},AbstractVector{Bool},KnetVector{Int32}}), (Union{Real,AbstractUnitRange,Colon}...) (in any order)
  * N-D: (Real...)

* Array operations: ==, !=, adjoint, argmax, argmin, cat, convert, copy, copyto!, deepcopy,
  display, eachindex, eltype, endof, fill!, findmax, findmin, first, hcat, isapprox,
  isempty, length, ndims, one, ones, permutedims, pointer, rand!, randn!, reshape, similar,
  size, stride, strides, summary, transpose, vcat, vec, zero.  (Boolean operators generate
  outputs with same type as inputs; no support for KnetArray{Bool}.)

* Unary functions with broadcasting: -, abs, abs2, acos, acosh, asin, asinh, atan, atanh,
  cbrt, ceil, cos, cosh, cospi, digamma, erf, erfc, erfcinv, erfcx, erfinv, exp, exp10,
  exp2, expm1, floor, gamma, lgamma, log, log10, log1p, log2, loggamma, one, round, sign,
  sin, sinh, sinpi, sqrt, tan, tanh, trigamma, trunc, zero

* Binary functions with broadcasting: !=, *, +, -, /, <, <=, ==, >, >=, ^, max, min

* Reduction operators: maximum, minimum, prod, sum
   
* Statistics: mean, std, stdm, var, varm

* Linear algebra: (*), axpy!, lmul!, norm, rmul!

* Knet extras: batchnorm, bce, bmm, cat1d, conv4, cpucopy, deconv4, dropout, elu, gpucopy,
  logistic, logp, logsoftmax, logsumexp, mat, nll, pool, relu, RNN, selu, sigm,
  softmax, unpool (Only 4D/5D, Float32/64 KnetArrays support conv4, pool, deconv4, unpool)

"""
mutable struct KnetArray{T,N} # <: AbstractArray{T,N} (TODO)
    ptr::KnetPtr
    dims::NTuple{N,Int}
end

# Note: I removed <: AbstractArray{T,N} after I painfully discovered some inefficient
# AbstractArray methods inherited unintentionally.  It is better to define a few extra
# methods to keep a tighter control on what methods exactly get called for KnetArrays.


# Aliases:
const KnetMatrix{T} = KnetArray{T,2}
const KnetVector{T} = KnetArray{T,1}
const KnetVecOrMat{T} = Union{KnetVector{T}, KnetMatrix{T}}
const DevArray{T,N} = Union{KnetArray{T,N},CuArray{T,N}}

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
KnetArray{T,N}(x::KnetArray{T,N}) where {T,N} = _unsafe_copy!(KnetArray{T}(undef,x.dims), 1, x, 1, prod(x.dims))
KnetArray{T,N}(x::KnetArray{S,N}) where {T,N,S} = _unsafe_copy!(KnetArray{T}(undef,x.dims), 1, convert(Array{T,N},x), 1, prod(x.dims))

# KnetArray(::AbstractArray)
KnetArray(A::AbstractArray{T,N})    where {T,N}   = KnetArray{T,N}(A)
KnetArray{T}(A::AbstractArray{S,N}) where {T,N,S} = KnetArray{T,N}(A)
KnetArray{T,N}(x::AbstractArray{S,N}) where {T,N,S} = _unsafe_copy!(KnetArray{T}(undef,size(x)), 1, convert(Array{T,N},x), 1, length(x))

# _unsafe_copy! does no bounds checking, the callers must. TODO: use a more standard method for this.
function _unsafe_copy!(dest::KnetArray{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int) where {T}
    cuMemcpyHtoD_v2(CuPtr{Nothing}(UInt(pointer(dest,doffs))), pointer(src,soffs), n*sizeof(T))
    return dest
end
function _unsafe_copy!(dest::Array{T}, doffs::Int, src::KnetArray{T}, soffs::Int, n::Int) where {T}
    cuMemcpyDtoH_v2(pointer(dest,doffs),CuPtr{Nothing}(UInt(pointer(src,soffs))), n*sizeof(T))
    return dest
end
function _unsafe_copy!(dest::KnetArray{T}, doffs::Int, src::KnetArray{T}, soffs::Int, n::Int) where {T}
    cuMemcpyDtoD_v2(CuPtr{Nothing}(UInt(pointer(dest,doffs))),CuPtr{Nothing}(UInt(pointer(src,soffs))), n*sizeof(T))
    return dest
end

# Conversions:

# Array(::KnetArray)
Array(A::KnetArray{T,N})    where {T,N}   = Array{T,N}(A)
Array{T}(A::KnetArray{S,N}) where {T,N,S} = Array{T,N}(A)
Array{T,N}(x::KnetArray{S,N}) where {T,N,S} = convert(Array{T,N}, _unsafe_copy!(Array{S}(undef,x.dims), 1, x, 1, prod(x.dims)))

# KnetArray <- KnetArray
convert(::Type{KnetArray}, x::KnetArray{T,N}) where {T,N} = x
convert(::Type{KnetArray{T}}, x::KnetArray{T,N}) where {T,N} = x
convert(::Type{KnetArray{T,N}}, x::KnetArray{T,N}) where {T,N} = x
convert(::Type{KnetArray{T}}, x::KnetArray{S,N}) where {T,N,S} = convert(KnetArray{T,N}, x)
convert(::Type{KnetArray{T,N}}, x::KnetArray{S,N}) where {T,N,S} = convert(KnetArray{T,N},_unsafe_copy!(Array{S,N}(undef,x.dims), 1, x, 1, prod(x.dims)))

# KnetArray <- AbstractArray
convert(::Type{KnetArray}, x::AbstractArray{T,N}) where {T,N} = convert(KnetArray{T,N}, x)
convert(::Type{KnetArray{T}}, x::AbstractArray{S,N}) where {T,N,S} = convert(KnetArray{T,N}, x)
convert(::Type{KnetArray{T,N}}, x::AbstractArray{S,N}) where {T,N,S} = _unsafe_copy!(KnetArray{T,N}(undef,size(x)), 1, convert(Array{T,N},x), 1, length(x))

# Array <- KnetArray
convert(::Type{Array}, x::KnetArray{T,N}) where {T,N} = convert(Array{T,N}, x)
convert(::Type{Array{T}}, x::KnetArray{S,N}) where {T,N,S} = convert(Array{T,N}, x)
convert(::Type{Array{T,N}}, x::KnetArray{S,N}) where {T,N,S} = convert(Array{T,N},_unsafe_copy!(Array{S}(undef,x.dims), 1, x, 1, prod(x.dims)))

# Ptr <- KnetArray

unsafe_convert(::Type{Ptr{T}}, a::KnetArray) where {T} = unsafe_convert(Ptr{T}, pointer(a))
pointer(a::KnetArray{T}) where {T} = convert(Ptr{T}, a.ptr.ptr)
pointer(a::KnetArray{T},i) where {T} = convert(Ptr{T}, a.ptr.ptr + (i-1)*sizeof(T))

function unsafe_convert(T::Type{<:CuPtr}, x::KnetArray)
    T(UInt(x.ptr.ptr))
end

# Extend function CuArray to create a memory shared CuArray from KnetArray:
# Avoid the cu function as it changes eltype to Float32
function CuArray(x::KnetArray{T}) where {T}
    p = CuPtr{T}(UInt(x.ptr.ptr))
    unsafe_wrap(CuArray{T}, p, x.dims; own=false)
end

function convert(A::Type{<:CuArray}, x::KnetArray)
    convert(A, CuArray(x))      # extra convert in case T,N changes
end

# Extend function KnetArray to create a memory shared KnetArray from CuArray:
function KnetArray(x::CuArray{T,N}) where {T,N}
    p = Base.bitcast(Cptr, pointer(x))
    k = KnetPtr(p, sizeof(x), Int(CUDA.device().handle), x) 
    KnetArray{T,N}(k, size(x))
end

function convert(A::Type{<:KnetArray}, x::CuArray)
    convert(A, KnetArray(x))    # extra convert in case T,N changes
end

function ka(x...)
    @warn "ka() is deprecated, please use KnetArray instead" maxlog=1
    KnetArray(x...)
end
