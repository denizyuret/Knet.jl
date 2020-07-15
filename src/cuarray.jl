using CUDA: CuPtr, unsafe_free!, usage_limit
import CUDA: CuArray

### Use CuArrays kernels as fallback for undefined KnetArray operations.

import Base: getindex, setindex!, permutedims, permutedims!, cat, hcat, vcat

# Extend function CuArray to create a memory shared CuArray from KnetArray:
# Avoid the cu function as it changes eltype to Float32
function CuArray(x::KnetArray{T}) where {T}
    p = CuPtr{T}(UInt(x.ptr.ptr))
    Base.unsafe_wrap(CuArray{T}, p, size(x); own=false)
end

# Based on _unsafe_getindex, multidimensional.jl:679, julia 1.2.0
function getindex(A::KnetArray, I...)
    I = Base.to_indices(A, I)
    shape = Base.index_shape(I...)
    B = similar(A, length.(shape))
    Base._unsafe_getindex!(CuArray(B), CuArray(A), I...)
    return B
end

function setindex!(A::KnetArray, B, I...)
    if B isa KnetArray || B isa AbstractArray
        B = CuArray(B)
    end
    setindex!(CuArray(A), B, I...)
    return A
end

permutedims!(y::KnetArray, x::KnetArray, perm) = (permutedims!(CuArray(y), CuArray(x), perm); y)

# Based on permutedims, multidimensional.jl:1334, julia 1.2.0
function permutedims(B::KnetArray,perm)
    dimsB = size(B)
    ndimsB = length(dimsB)
    (ndimsB == length(perm) && isperm(perm)) || throw(ArgumentError("no valid permutation of dimensions"))
    dimsP = ntuple(i->dimsB[perm[i]], ndimsB)::typeof(dimsB)
    P = similar(B, dimsP)
    permutedims!(P,B,perm)
end

#permutedims(x::KnetMatrix)=permutedims(x,(2,1))  # CUDA.jl is %10 faster but has startup cost
permutedims(x::KnetMatrix)=_transpose(x)          # cuDNN is %10 slower but no startup cost
permutedims(x::KnetVector)=copy(reshape(x,1,:))

using Base: dims2cat, cat_shape, __cat

# vcat(X::KnetArray...)=cat(X...; dims=Val(1)) # karray.jl version is 30%-80% faster
hcat(X::KnetArray...)=cat(X...; dims=Val(2))   # This should only kick in for dims > 2, karray.jl 1/2D versions are 100% faster

# Based on _cat_t, abstractarray.jl:1439, julia 1.2.0
function cat(X::KnetArray{T}...; dims) where {T}
    catdims = dims2cat(dims)
    # catdims == (true,) && return vcat_old(X...) # 10-30% faster
    shape = cat_shape(catdims, (), map(size, X)...)
    # length(shape) <= 2 && catdims == (false,true) && return hcat_old(X...) # 50% faster
    A = similar(X[1], T, shape) # cat_similar(X[1], T, shape)
    if T <: Number && count(!iszero, catdims) > 1
        fill!(A, zero(T))
    end
    __cat(CuArray(A), shape, catdims, map(CuArray,X)...)
    return A
end

# Must be careful with memory management, for now we will let Knet manage memory.
# use CuArray(x) with overwriting kernels only.
# use the following with caution.

if has_cuda()
    function KnetArray(x::CuArray{T,N}) where {T,N}
        p = Base.bitcast(Knet.Cptr, x.ptr)
        k = Knet.KnetPtr(p, sizeof(x), gpu(), x) 
        KnetArray{T,N}(k, size(x))
    end
end

# Testing the CUDA.jl allocator: set Knet.cuallocator()=true to use this
function KnetPtrCu(len::Int)
    c = CuArray{UInt8}(undef, len)
    p = convert(Cptr, convert(UInt, Base.unsafe_convert(CuPtr{UInt8}, Base.cconvert(CuPtr{UInt8}, c))))
    kp = KnetPtr(p, len, gpu(), c)
    finalizer(freeKnetPtrCu, kp)
end

function freeKnetPtrCu(p::KnetPtr)
    # GC.gc comes here directly, manual calls come through freeKnetPtr()
    # After a manual call, GC.gc may call the finalizer again, avoid double free
    if p.parent isa CuArray
        unsafe_free!(p.parent)
        p.ptr, p.parent = C_NULL, nothing
    elseif p.parent isa Nothing
        @assert p.ptr == C_NULL
        # already freed, do nothing
    elseif p.parent isa KnetPtr
        # subarray, do nothing
    else
        error("Bad parent pointer $(typeof(p.parent))")
    end
end

# argmax, argmin etc. Fixes https://github.com/denizyuret/Knet.jl/issues/368.
# Two options: argmax(Array(KnetArray)) vs argmax(CuArray)
# Experiments: 10x10, 100x100, 1000x1000 with no dims, dims=1, dims=2
# With no dims, CUDA.jl is better for 100x100, 1000x1000.
# With all others, KnetArray is better.

import Base: argmax, argmin, findmax, findmin
# TODO: try this again after https://github.com/JuliaGPU/CuArrays.jl/issues/304 is resolved
# argmaxarray(x,d)=((d===:) && length(x) > 4096 ? CuArray(x) : Array(x))
argmaxarray(x,d)=Array(x)
argmax(x::KnetArray; dims=:)=argmax(argmaxarray(x,dims); dims=dims)
argmin(x::KnetArray; dims=:)=argmin(argmaxarray(x,dims); dims=dims)
findmax(x::KnetArray; dims=:)=findmax(argmaxarray(x,dims); dims=dims)
findmin(x::KnetArray; dims=:)=findmin(argmaxarray(x,dims); dims=dims)


# Issue #108:Element-wise power of KnetArray give NaN results #108
# This is a bug with CUDA giving NaN for integer powers of negative numbers (powf is broken)

import Base.Broadcast: broadcasted

function broadcasted(::typeof(^),a::KnetArray{T},s::Number) where T
    b = similar(a)
    ca = CuArray(a)
    cb = CuArray(b)
    cb .= ca .^ T(s)
    return b
end

function broadcasted(::typeof(^),s::Number,a::KnetArray{T}) where T
    b = similar(a)
    ca = CuArray(a)
    cb = CuArray(b)
    cb .= T(s) .^ ca
    return b
end


########################################################################
# AutoGrad functions extended for CuArray (copied from karray.jl)

# using CUDA: CuArray, CuPtr
# using Knet: c2i, @knet8
using Base.Broadcast: Broadcasted
using AutoGrad: Value, recording

import AutoGrad: Sparse, matches, ungetindex, addto!, addtoindex!, zeroslike
import Base: copyto!
import LinearAlgebra: axpy!

Sparse(a::CuArray{T,N},v,i) where {T,N} = Sparse{T,N}(a,v,i)

axpy!(a, x::Sparse, y::CuArray) = addto!(y, a*x)

function copyto!(a::CuArray, bc::Broadcasted{S,A,F,X}) where
    {S, A, F <: Union{typeof(+),typeof(-)}, X <: Tuple{Any,Sparse}}
    (b,c) = bc.args
    if !(size(a) == size(b) == size(c.container))
        a .= bc.f.(b, full(c))
        return a
    end
    a === b || copyto!(a, b)
    F <: typeof(-) && (c = -c)
    addto!(a, c)
    return a
end

matches(a::CuArray,b::CuArray)=(size(a)==size(b))

function addto!(a::CuArray{T},b::CuArray{T}) where {T}
    if AutoGrad.recording(); a = copy(a); end  # support highorder gradients
    axpy!(1,b,a) # (a+b)
end

function addto!(a::CuArray,b::Sparse)
    @assert size(a) == size(b.container)
    if AutoGrad.recording(); a = copy(a); end  # support highorder gradients
    for (idx,val) in zip(b.indices, b.values)
        addtoindex!(a, val, idx...)
    end
    return a
end

addto!(a::Sparse, b::CuArray) = addto!(b, a)

import Base: +, -
+(a::CuArray, s::Sparse) = addto!(copy(a), s)
+(s::Sparse, a::CuArray) = addto!(copy(a), s)
-(a::CuArray, s::Sparse) = addto!(copy(a), -s)
-(s::Sparse, a::CuArray) = addto!(-a, s)

function ungetindex(x::CuArray{T},dxi,i) where T
    if isbitstype(T)
        if dxi isa Value
            forw(addto!, zeroslike(x), forw(ungetindex, x, dxi, i))
        elseif recording()
            addtoindex!(zero(x), dxi, i...)
        else
            Sparse(x,[dxi],[i])
        end
    else
        # Using addtoindex! instead of setindex! to handle repeated indices
        addtoindex!(Array{Union{T,Nothing}}(nothing, size(x)), dxi, i...)
    end
end

zeroslike(a::CuArray)=zero(a) # Still need this because zero(::Array{!isbits}) is not defined


# This only works when there are no repeated indices. This is true for index types:
# Real, (Real...), CartesianIndex, Colon, AbstractArray{Bool}, Range, EmptyArray
# and pairs of Union{Real,AbstractUnitRange,Colon} and (Colon,Range)
addtoindex!(A::CuArray, X, I...)=setindex!(A, getindex(A,I...) .+ X, I...)

# The following index types may have repeated indices:
# AbstractArray{Real}, AbstractArray{CartesianIndex}, (Colon,AbstractVector{Real}), (AbstractVector{Real},Colon)

addtoindex!(A::CuArray, X, I::AbstractArray{T}) where {T<:CartesianIndex}=addtoindex!(A,X,c2i(size(A),I))

for F in (32,64); T=Symbol("Float$F"); @eval begin

    function addtoindex!(A::CuArray{$T}, X, I::AbstractArray{R}) where {R<:Real}
        I = CuArray{Int32}(I)
        X = CuArray{$T}(X)
        @knet8($("addents_$F"),(Cint,CuPtr{Int},CuPtr{$T},CuPtr{$T}),
               length(I), I, A, X)
        return A
    end

    function addtoindex!(A::CuArray{$T}, X, ::Colon, I::AbstractArray{R}) where {R<:Real}
        I = CuArray{Int32}(I)
        X = CuArray{$T}(X)
        @knet8($("addcols_$F"),(Cint,Cint,Cint,CuPtr{Int},CuPtr{$T},CuPtr{$T}),
               size(A,1), size(A,2), length(I), I, A, X)
        return A
    end

    function addtoindex!(A::CuArray{$T}, X, I::AbstractArray{R}, ::Colon) where {R<:Real}
        I = CuArray{Int32}(I)
        X = CuArray{$T}(X)
        @knet8($("addrows_$F"),(Cint,Cint,Cint,CuPtr{Int},CuPtr{$T},CuPtr{$T}),
               size(A,1), size(A,2), length(I), I, A, X)
        return A
    end

    addtoindex!(A::CuArray{$T}, X, I::AbstractArray{Bool})=addtoindex!(A,X,findall(vec(I)))
    addtoindex!(A::CuArray{$T}, X, c::Colon, I::AbstractArray{Bool})=addtoindex!(A,X,c,findall(vec(I)))
    addtoindex!(A::CuArray{$T}, X, I::AbstractArray{Bool}, c::Colon)=addtoindex!(A,X,findall(vec(I)),c)

end; end
