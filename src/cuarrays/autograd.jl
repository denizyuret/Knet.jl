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
