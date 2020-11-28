import Base: getindex, setindex!
import Base.Broadcast: materialize!
using Base.Broadcast: Broadcasted
using CUDA: CuArray, CuPtr, cuMemcpyDtoD_v2
using Knet.LibKnet8: @knet8
# include("kptr.jl") ## KnetPtr, Cptr
# include("karray.jl") ## KnetArray, KnetMatrix, KnetVector

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


# CuArray fallbacks: these rely on KnetArray<->CuArray conversions having shared pointers
# and cover the cases which are not explicitly defined using kernels below.

# Fallback for A[I...]  => getindex(A, I...)
function getindex(A::KnetArray, I...)
    _A = CuArray(A)
    _B = getindex(_A, I...)
    B = (_B isa CuArray ? KnetArray(_B) : _B)
    return B
end

# Fallback for A[I...] = B  => setindex!(A, B, I...)
function setindex!(A::KnetArray, B, I...)
    _A = CuArray(A)
    _B = (B isa KnetArray || B isa AbstractArray ? CuArray(B) : B)
    setindex!(_A, _B, I...)  ## This only works for x[I...] = y but not for x[I...] .= y
    return A
end

# Fallback for A[I...] .= B  => materialize!(dotview(A, I...), broadcasted(identity, B))
function materialize!(A::SubArray{T,N,<:KnetArray}, B) where {T,N}
    _A = view(CuArray(A.parent), A.indices...)
    _B = (B isa KnetArray || B isa AbstractArray ? CuArray(B) : B)
    materialize!(_A, _B)
end

# For contiguous I, dotview(A, I...) gives a shared-memory KnetArray rather than a view
function materialize!(A::KnetArray, B) where {T,N}
    _A = CuArray(A)
    _B = (B isa KnetArray || B isa AbstractArray ? CuArray(B) : B)
    materialize!(_A, _B)
end

# Ambiguity fixes:
function materialize!(A::SubArray{T,N,<:KnetArray}, B::Broadcasted{S}) where {T,N,S}
    _A = view(CuArray(A.parent), A.indices...)
    materialize!(_A, B)
end

function materialize!(A::KnetArray{T,N}, B::Broadcasted{S}) where {T,N,S}
    _A = CuArray(A)
    materialize!(_A, B)
end


# The following fallback version tried to do all allocations using KnetArrays but was recently broken (Issue 618).
# This only makes a difference if we are using the Knet allocator (i.e. cuallocator[] = false) which is not the default any more.
# Based on julia-1.4.2/base: getindex@abstractarray.jl:980, _getindex@multidimensional.jl:726, _unsafe_getindex!@multidimensional.jl:738
# function getindex_broken(A::KnetArray, I...)
#     _A = CuArray(A)
#     I = Base.to_indices(_A, I)
#     checkbounds(_A, I...)
#     shape = Base.index_shape(I...)
#     B = similar(A, length.(shape))
#     _B = CuArray(B)
#     Base._unsafe_getindex!(_B, _A, I...)
#     return B
# end


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
    return A
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
    return A
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
        return a
    end
end

function setindex!(A::KnetArray{T}, v::Real, I::AbstractUnitRange) where {T}
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    if length(I)==0; return A; end
    unsafe_setindex!(A,T(v),I)
    return A
end

function setindex!(A::KnetArray{T}, v::Real, I::AbstractUnitRange{Bool}) where {T} # julia4 ambig fix
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    if length(I)==0; return A; end
    unsafe_setindex!(A,T(v),I)
    return A
end

function setindex!(A::KnetArray{T}, v, I::AbstractUnitRange) where {T}
    if !(1 <= first(I) <= last(I) <= length(A)); throw(BoundsError(A,I)); end
    if length(v)!=length(I); throw(DimensionMismatch()); end
    if length(I)==0; return A; end
    if eltype(v)!=T; v = convert(Array{T},v); end
    _unsafe_copy!(A,first(I),v,1,length(I))
    return A
end

## Indexing with Colon
# Note that getindex(a,:) returns a view not a copy

function getindex(A::KnetArray, I::Colon)
    reshape(A,length(A))
end

function setindex!(A::KnetArray{T}, v::Real, I::Colon) where {T}
    if length(A)==0; return A; end
    unsafe_setindex!(A, T(v), 1:length(A))
    return A
end

function setindex!(A::KnetArray{T}, v, I::Colon) where {T}
    if length(v)!=length(A); throw(DimensionMismatch()); end
    if length(v)==0; return A; end
    if eltype(v)!=T; v = convert(Array{T},v); end
    _unsafe_copy!(A,1,v,1,length(A))
    return A
end

for F in (32,64); T=Symbol("Float$F"); @eval begin

## Indexing with KnetArray{Int32}: low level, only Int32 supported, no bounds checking

    function unsafe_getindex!(x::KnetArray{$T}, y::KnetArray{$T}, i::KnetArray{Int32})
        @knet8($("getents_$F"),(Cint,Ptr{Int},Ptr{$T},Ptr{$T}), length(i), i, x, y)
        return y
    end

    function unsafe_setindex!(x::KnetArray{$T}, y::$T, i::KnetArray{Int32})
        @knet8($("setent1_$F"),(Cint,Ptr{Int},Ptr{$T},$T), length(i), i, x, y)
        return x
    end

    function unsafe_setindex!(x::KnetArray{$T}, y::KnetArray{$T}, i::KnetArray{Int32})
        @knet8($("setents_$F"),(Cint,Ptr{Int},Ptr{$T},Ptr{$T}), length(i), i, x, y)
        return x
    end

## Indexing with (Colon,KnetArray{Int32})
# TODO: Just special case rows and columns in matrices until we have a more general solution

    function unsafe_getindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, ::Colon, i::KnetVector{Int32})
        @knet8($("getcols_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(x,1), size(x,2), length(i), i, x, y)
        return y
    end

    function unsafe_setindex!(x::KnetMatrix{$T}, y::$T, ::Colon, i::KnetVector{Int32})
        @knet8($("setcol1_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},$T),
               size(x,1), size(x,2), length(i), i, x, y)
        return x
    end

    function unsafe_setindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, ::Colon, i::KnetVector{Int32})
        @knet8($("setcols_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(x,1), size(x,2), length(i), i, x, y)
        return x
    end

## Indexing with (KnetArray{Int32},Colon)

    function unsafe_getindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, i::KnetVector{Int32}, ::Colon)
        @knet8($("getrows_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(x,1), size(x,2), length(i), i, x, y)
        return y
    end

    function unsafe_setindex!(x::KnetMatrix{$T}, y::KnetMatrix{$T}, i::KnetVector{Int32}, ::Colon)
        @knet8($("setrows_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},Ptr{$T}),
               size(x,1), size(x,2), length(i), i, x, y)
        return x
    end

    function unsafe_setindex!(x::KnetMatrix{$T}, y::$T, i::KnetVector{Int32}, ::Colon)
        @knet8($("setrow1_$F"),(Cint,Cint,Cint,Ptr{Int},Ptr{$T},$T),
               size(x,1), size(x,2), length(i), i, x, y)
        return x
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

# function setindex!(x::KnetMatrix{T}, y, c::AbstractUnitRange, i::AbstractVector{I}) where {T,I<:Real}
#     if c == 1:size(x,1)
#         setindex!(x, y, :, i)
#     else
#         throw(MethodError(setindex!,x,y,c,i))
#     end
# end

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

# function setindex!(x::KnetMatrix{T}, y, i::AbstractVector{I}, c::AbstractUnitRange) where {T,I<:Real}
#     if c == 1:size(x,2)
#         setindex!(x, y, i, :)
#     else
#         throw(MethodError(setindex!,x,y,i,c))
#     end
# end

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

# function setindex!(A::KnetMatrix, v, I::StepRange, r::AbstractUnitRange)
#     if r == 1:size(A,2)
#         setindex!(A,v,I,:)
#     else
#         throw(MethodError(setindex!, A, v, I, r))
#     end
# end

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

# function setindex!(A::KnetMatrix, v, r::AbstractUnitRange, I::StepRange)
#     if r == 1:size(A,1)
#         setindex!(A,v,:,I)
#     else
#         throw(MethodError(setindex!, A, v, r, I))
#     end
# end

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

# function setindex!(x::KnetMatrix, y, c::AbstractUnitRange, i::AbstractVector{Bool})
#     if c == 1:size(x,1)
#         setindex!(x,y,:,i)
#     else
#         throw(MethodError(setindex!,x,y,c,i))
#     end
# end

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

# function setindex!(x::KnetMatrix, y, i::AbstractVector{Bool}, c::AbstractUnitRange)
#     if c == 1:size(x,2)
#         setindex!(x,y,i,:)
#     else
#         throw(MethodError(setindex!,x,y,i,c))
#     end
# end

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
                # @cudart(cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),
                #         aptr0, B, nelts*sizeof(T), cudadir(A,B))
                cuMemcpyDtoD_v2(CuPtr{Nothing}(UInt(aptr0)), CuPtr{Nothing}(UInt(pointer(B))), nelts*sizeof(T))
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

# function setindex!(x::KnetArray, y, i::AbstractUnitRange, j::AbstractUnitRange, k::Index3)
#     if first(i) == 1 && last(i) == size(x,1) && first(j) == 1 && last(j) == size(x,2)
#         setindex!(x, y, :, :, k)
#     else
#         throw(MethodError(setindex!, (x,y,i,j,k)))
#     end
# end

function getindex(x::KnetArray{T,2}, ::Colon, m::AbstractArray{I,2}) where {T,I<:Integer}
    reshape(x[:,vec(m)], size(x,1), size(m,1), size(m,2))
end


