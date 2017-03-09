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

* Array operations: ==, !=, cat, convert, copy, copy!, deepcopy,
  display, eachindex, eltype, endof, fill!, first, getindex, hcat,
  isapprox, isempty, length, linearindexing, ndims, ones, pointer,
  rand!, reshape, setindex!, similar, size, stride, strides, summary,
  vcat, vec, zeros.  (Only Integer, Colon, and UnitRange indices
  supported for get/setindex.  CartesianIndex, StepRange, Array, and
  Bool indices not supported.  cat(i,x,y) supported for i=1,2.)

* Math operators: (-), abs, abs2, acos, acosh, asin, asinh, atan,
  atanh, cbrt, ceil, cos, cosh, cospi, erf, erfc, erfcinv, erfcx,
  erfinv, exp, exp10, exp2, expm1, floor, log, log10, log1p, log2,
  round, sign, sin, sinh, sinpi, sqrt, tan, tanh, trunc

* Broadcasting operators: (.*), (.+), (.-), (./), (.<), (.<=), (.!=),
  (.==), (.>), (.>=), (.^), max, min.  (Only Array-Scalar and
  Array-Vector broadcasting are supported. Boolean operators generate
  outputs with same type as inputs; no support for KnetArray{Bool}.)

* Reduction operators: countnz, maximum, minimum, prod, sum, sumabs,
  sumabs2, vecnorm.  (Only Array->Scalar and Array->Vector reductions
  are supported)
    
* Linear algebra: (*), axpy!, permutedims (only 2D and 3D), transpose

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
typealias KnetMatrix{T} KnetArray{T,2}
typealias KnetVector{T} KnetArray{T,1}
typealias KnetVecOrMat{T} Union{KnetVector{T}, KnetMatrix{T}}

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
convert{T,N,S}(::Type{KnetArray{T,N}}, x::KnetArray{S,N}) = convert(KnetArray{T,N},unsafe_copy!(Array(S, size(x)), 1, x, 1, length(x)))
reshape{T}(a::KnetArray{T},dims::Dims)=(if dims==size(a); a; elseif prod(dims)!=length(a); throw(DimensionMismatch()); else; KnetArray{T,length(dims)}(a.ptr,dims); end)
reshape(a::KnetArray, dims::Int...) = reshape(a, dims)
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
import Base: eachindex, eltype, endof, fill!, first, isempty, length, linearindexing, ndims, ones, similar, size, stride, strides, zeros, (==), isapprox
eachindex(a::KnetArray) = (1:length(a))
eltype{T}(::KnetArray{T})=T
eltype{T}(::Type{KnetArray{T}}) = T
eltype{T,n}(::Type{KnetArray{T,n}}) = T
endof(a::KnetArray) = length(a)
fill!{T}(a::KnetArray{T},x)=(knetfill!(a,T(x),1,length(a));a)
first(a::KnetArray) = a[1]
# AutoGrad leaves `first` as a compound proc calling start which doesn't work with KnetArrays
@primitive  first(x::KnetArray),dy,y  AutoGrad.ungetindex(x,dy,1)
isempty(a::KnetArray) = (0==length(a))
length(a::KnetArray)=prod(size(a))
linearindexing(::KnetArray)=Base.LinearFast()
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


# Indexing:
import Base: getindex, setindex!

# We will implement indexing ranges as views not copies, if possible (when contiguous).
# For contiguous memory without stride all but the last >1 dimension must be full

# The original getindex(a,i:j...) for AbstractArray copies:
# function _getindex(l::LinearIndexing, A::AbstractArray, I::Union{Real, AbstractArray, Colon}...)
# in abstractarray.jl:487,multidimensional.jl:184.

# which getindex ops does array implement?
# getindex(A::Array, i1::Real)
# getindex(A::Array, i1::Real, i2::Real, I::Real...)
# getindex(A::Array, I::UnitRange{Int})
# getindex(A::Array, c::Colon)
# getindex{T<:Real}(A::Array, I::Range{T})

# Julia #14770
# If I is shorter than ndims(A) but longer than 1 the remaining indices assumed =1
# Also extra 1's at the end of I are ignored

# These two are not sufficient in spite of what the documentation says:
# display goes into an infinite loop!
# getindex{T}(A::KnetArray{T}, i::Int)=unsafe_copy!(T[0], 1, A, i, 1)[1]
# setindex!{T}(A::KnetArray{T}, v, i::Int)=unsafe_copy!(A, i, T[v], 1, 1)

# First deal with the easy cases: integer indices, a Colon or a UnitRange.

function getindex{T}(A::KnetArray{T}, I::Real)
    J = Int(I)
    1 <= J <= length(A) || throw(BoundsError(A,J))
    unsafe_copy!(T[0], 1, A, J, 1)[1]
end

function setindex!{T}(A::KnetArray{T}, v, I::Real)
    J = Int(I)
    1 <= J <= length(A) || throw(BoundsError(A,J))
    unsafe_copy!(A, J, T[v], 1, 1)
end

function getindex{T}(A::KnetArray{T}, I::Real...)
    J = Base.to_indexes(I...)
    @inbounds for j=1:length(J)
        1 <= J[j] <= size(A,j) || throw(BoundsError(A,J))
    end
    i = sub2ind(size(A), J...)
    unsafe_copy!(T[0], 1, A, i, 1)[1]
end

function setindex!{T}(A::KnetArray{T}, v, I::Real...)
    J = Base.to_indexes(I...)
    @inbounds for j=1:length(J)
        1 <= J[j] <= size(A,j) || throw(BoundsError(A,J))
    end
    i = sub2ind(size(A), J...)
    unsafe_copy!(A, i, T[v], 1, 1)
end

function getindex{T}(A::KnetArray{T}, I::UnitRange)
    1 <= first(I) <= last(I) <= length(A) || throw(BoundsError(A,I))
    off = 1+(first(I)-1)*sizeof(T)
    len = length(I)*sizeof(T)
    ptr = KnetPtr(A.ptr, off, len)
    KnetArray{T,1}(ptr, (length(I),))
end

function setindex!{T}(A::KnetArray{T}, v, I::UnitRange)
    1 <= first(I) <= last(I) <= length(A) || throw(BoundsError(A,I))
    if isa(v,Number)
        knetfill!(A,T(v),first(I),length(I))
    elseif (isa(v,KnetArray) || isa(v,Array))
        length(v)==length(I) || throw(DimensionMismatch())
        eltype(v)==T || (v = convert(Array{T},v))
        unsafe_copy!(A,first(I),v,1,length(I))
    else
        throw(MethodError(setindex!, A, v, I))
    end
end

function getindex(A::KnetArray, I::Colon)
    reshape(A,length(A))
end

function setindex!{T}(A::KnetArray{T}, v, I::Colon)
    if isa(v,Number)
        knetfill!(A,T(v),1,length(A))
    elseif (isa(v,KnetArray) || isa(v,Array))
        length(v)==length(A) || throw(DimensionMismatch())
        eltype(v)==T || (v = convert(Array{T},v))
        unsafe_copy!(A,1,v,1,length(A))
    else
        throw(MethodError(setindex!, A, v, I))
    end
end

# TODO: the following getindex, setindex! work for 1 and 2 dimensions only, write general versions.

function getindex{T,N}(A::KnetArray{T,N}, I::Union{Real, UnitRange, Colon}...)
    (nelts,nrows,ncols,firstindex,astep) = indexparams(A,I...)
    B1 = isa(I[1],Colon) ? size(A,1) : length(I[1])
    B2 = isa(I[2],Colon) ? size(A,2) : length(I[2])
    if ncols == 1
        off = 1+(firstindex-1)*sizeof(T)
        len = nrows*sizeof(T)
        ptr = KnetPtr(A.ptr, off, len)
        KnetArray{T,2}(ptr, (B1,B2))
    else
        B = similar(A, (B1,B2))
        nrows *= sizeof(T); astep *= sizeof(T)
        ccall((:xcopy,libknet8),Void,(Cint,Cint,Cptr,Cint,Cptr,Cint),
              nrows, ncols, pointer(A,firstindex), astep, B, nrows)
        return B
    end
end

function setindex!{T,N}(A::KnetArray{T,N}, B, I::Union{Real, UnitRange, Colon}...)
    (nelts,nrows,ncols,firstindex,astep) = indexparams(A,I...)
    aptr0 = pointer(A, firstindex)
    if isa(B,Number)
        B = T(B)
        if ncols == 1
            if T <: Float32;    ccall((:fill_32,libknet8),Void,(Cint,Cfloat, Ptr{Cfloat}), nelts,B,aptr0)
            elseif T<: Float64; ccall((:fill_64,libknet8),Void,(Cint,Cdouble,Ptr{Cdouble}),nelts,B,aptr0)
            else error("$T not supported"); end
        else
            if T <: Float32;    ccall((:xfill_32,libknet8),Void,(Cint,Cint,Cfloat, Ptr{Cfloat}, Cint),nrows,ncols,B,aptr0,astep)
            elseif T<: Float64; ccall((:xfill_64,libknet8),Void,(Cint,Cint,Cdouble,Ptr{Cdouble},Cint),nrows,ncols,B,aptr0,astep)
            else error("$T not supported"); end
        end
    else
        length(B) == nelts || throw(DimensionMismatch())
        B = convert(KnetArray{T},B)
        if ncols == 1
            @cuda(cudart,cudaMemcpyAsync,(Cptr,Cptr,Csize_t,UInt32,Cptr),
                  aptr0, B, nelts*sizeof(T), cudadir(A,B), C_NULL)
        else
            nrows *= sizeof(T); astep *= sizeof(T)
            ccall((:xcopy,libknet8),Void,(Cint,Cint,Cptr,Cint,Cptr,Cint), nrows, ncols, B, nrows, aptr0, astep)
        end
    end
    return A
end

function indexparams{T,N}(A::KnetArray{T,N}, I::Union{Real, UnitRange, Colon}...)
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


# Just special case rows and columns until we have a more general solution
# for Array{Int} indices

for F in (32,64); T=Symbol("Float$F"); @eval begin

    function getindex(x::KnetMatrix{$T}, ::Colon, i::KnetVector{Int32})
        y = similar(x, size(x,1), length(i))
        ccall(($("getcols_$F"),libknet8),Void,(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
              size(x,1), size(x,2), length(i), i, x, y)
        return y
    end

    function getindex(x::KnetMatrix{$T}, i::KnetVector{Int32}, ::Colon)
        y = similar(x, length(i), size(x,2))
        ccall(($("getrows_$F"),libknet8),Void,(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
              size(x,1), size(x,2), length(i), i, x, y)
        return y
    end

    function setindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, ::Colon, i::KnetVector{Int32})
        ccall(($("setcols_$F"),libknet8),Void,(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
              size(x,1), size(x,2), length(i), i, x, y)
        return x
    end

    function setindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, i::KnetVector{Int32}, ::Colon)
        ccall(($("setrows_$F"),libknet8),Void,(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
              size(x,1), size(x,2), length(i), i, x, y)
        return x
    end

end; end

function getindex{T,I<:Integer}(x::KnetMatrix{T}, c::Colon, i::Vector{I})
    all(1 .<= i .<= size(x,2)) || throw(BoundsError(x,i))
    getindex(x,c,KnetArray{Int32}(i))
end
function getindex{T,I<:Integer}(x::KnetMatrix{T}, i::Vector{I}, c::Colon)
    all(1 .<= i .<= size(x,1)) || throw(BoundsError(x,i))
    getindex(x,KnetArray{Int32}(i),c)
end
function setindex!{T,I<:Integer}(x::KnetMatrix{T}, y::KnetMatrix{T}, c::Colon, i::Vector{I})
    size(x,1)==size(y,1) || throw(DimensionMismatch())
    all(1 .<= i .<= size(x,2)) || throw(BoundsError(x,i))
    setindex!(x,y,c,KnetArray{Int32}(i))
end
function setindex!{T,I<:Integer}(x::KnetMatrix{T}, y::KnetMatrix{T}, i::Vector{I}, c::Colon)
    size(x,2)==size(y,2) || throw(DimensionMismatch())
    all(1 .<= i .<= size(x,1)) || throw(BoundsError(x,i))
    setindex!(x,y,KnetArray{Int32}(i),c)
end


# Concatenation:
import Base: hcat, vcat, cat

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

function hcat{T}(a::KnetVecOrMat{T}, b::KnetVecOrMat{T})
    size(a,1)==size(b,1) || throw(DimensionMismatch())
    c1 = size(a,1)
    c2 = size(a,2) + size(b,2)
    c = KnetArray(T, (c1,c2))
    c[:,1:size(a,2)] = a
    c[:,1+size(a,2):end] = b
    return c
end

function vcat{T}(a::KnetVector{T}, b::KnetVector{T})
    c = KnetArray(T, length(a)+length(b))
    c[1:length(a)] = a
    c[1+length(a):end] = b
    return c
end

function vcat{T}(a::KnetVecOrMat{T}, b::KnetVecOrMat{T})
    size(a,2)==size(b,2) || throw(DimensionMismatch())
    c1 = size(a,1) + size(b,1)
    c2 = size(a,2)
    c = KnetArray(T, (c1,c2))
    c[1:size(a,1),:] = a
    c[1+size(a,1):end,:] = b
    return c
end

function cat{T}(d, a::KnetVecOrMat{T}, b::KnetVecOrMat{T})
    if     d==1; vcat(a,b)
    elseif d==2; hcat(a,b)
    else error("cat($d) not implemented.")
    end
end


# Utilities:

# Generalizing low level copy using linear indexing to/from gpu arrays:
# copy!{T}(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer)
# Note that this is an unsafe operation, no argument or bounds checking performed.
# Defined in Base:
# unsafe_copy!{T}(dest::Ptr{T}, src::Ptr{T}, n) at array.jl:73
# unsafe_copy!{T}(dest::Array{T,N}, doffs, src::Array{T,N}, soffs, n) at array.jl:79

import Base: unsafe_copy!, copy, copy!
typealias KorA{T} Union{KnetArray{T},Array{T}}

function copy!{T}(dest::KorA{T}, doffs::Integer, src::KorA{T}, soffs::Integer, n::Integer; stream=C_NULL)
    n == 0 && return dest
    n > 0 || throw(ArgumentError(string("tried to copy n=", n, " elements, but n should be nonnegative")))
    if soffs < 1 || doffs < 1 || soffs+n-1 > length(src) || doffs+n-1 > length(dest)
        throw(BoundsError())
    end
    unsafe_copy!(dest, doffs, src, soffs, n; stream=stream)
end

copy!{T}(dest::KorA{T}, src::KorA{T}) = copy!(dest, 1, src, 1, length(src))

copy(a::KnetArray)=unsafe_copy!(similar(a),1,a,1,length(a))

# This will make deepcopy work properly
Base.deepcopy_internal(x::KnetArray, s::ObjectIdDict)=if haskey(s,x); s[x]; else; copy(x); end

function unsafe_copy!{T}(dest::KorA{T}, doffs, src::KorA{T}, soffs, n; stream=C_NULL)
    @cuda(cudart,cudaMemcpyAsync,(Cptr,Cptr,Csize_t,UInt32,Cptr),
          pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), cudadir(dest,src), stream)
    return dest
end

function cudadir(a,b)
    deva = isa(a,KnetArray) && a.ptr.dev >= 0
    devb = isa(b,KnetArray) && b.ptr.dev >= 0
    if !deva && !devb; return 0
    elseif deva && !devb; return 1
    elseif !deva && devb; return 2
    elseif deva && devb;  return 3
    end
end

# Efficient fill:
for S in (32,64); T = Symbol("Float$S"); F = "fill_$S"
    @eval function knetfill!(a::KnetArray{$T},v::$T,off,len)
        ccall(($F,$libknet8),Void,(Cint,$T,Ptr{$T}),len,v,pointer(a,off))
    end
end

# AutoGrad functions:
import AutoGrad: zeroslike, sum_outgrads, UngetIndex, unary_nd, indexed_function, isequivalent, _dbg, ssize
zeroslike(a::KnetArray)=zeros(a)
sum_outgrads{T}(a::KnetArray{T},b::KnetArray{T})=(a+b)
sum_outgrads(a::KnetArray,b::UngetIndex)=setindex!(a,sum_outgrads(getindex(a,b.index...),b.value),b.index...)
unary_nd(f, x::KnetArray, eps) = reshape(eltype(x)[unary_nd(indexed_function(f, x, i), x[i], eps) for i in 1:length(x)], size(x))
isequivalent(x::Union{KnetArray,AbstractArray}, y::Union{KnetArray,AbstractArray}; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))
_dbg(a::KnetArray) = "K"*_dbg(a[1])*ssize(a)

# Hack for printing without copying the whole KnetArray and without inheriting AbstractArray:
import Base: display, summary
type KnetDisplay{T,N} <: AbstractArray{T,N}; a::KnetArray{T,N}; end
getindex(a::KnetDisplay, i...) = getindex(a.a, i...)
size(a::KnetDisplay) = size(a.a)
summary(a::KnetDisplay) = summary(a.a)
summary(a::KnetArray) = string(Base.dims2string(size(a)), " ", typeof(a))
display(a::KnetArray) = display(KnetDisplay(a))

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
