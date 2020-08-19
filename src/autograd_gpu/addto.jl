import AutoGrad: addto!, addtoindex!, matches
using CUDA: CuArray, CuPtr
using Knet.KnetArrays: KnetArray, DevArray, Cptr
using Knet.LibKnet8: @knet8
using AutoGrad: Sparse, recording

function addto!(a::DevArray{T},b::DevArray{T}) where {T}
    if recording(); a = copy(a); end  # support highorder gradients
    axpy!(1,b,a) # (a+b)
end

function addto!(a::DevArray,b::Sparse)
    @assert matches(a, b.container)
    if recording(); a = copy(a); end  # support highorder gradients
    for (idx,val) in zip(b.indices, b.values)
        addtoindex!(a, val, idx...)
    end
    return a
end

addto!(a::Sparse, b::DevArray) = addto!(b, a)

matches(a::DevArray,b::DevArray)=(typeof(a)===typeof(b) && size(a)===size(b))


# Setindex! works for addtoindex! when there are no repeated indices. This is true for index types:
# Real, (Real...), CartesianIndex, Colon, AbstractArray{Bool}, Range, EmptyArray
# and pairs of Union{Real,AbstractUnitRange,Colon} and (Colon,Range)

# AutoGrad defines the following 3 methods for addtoindex!
# This is most specific, catches cases of I::Union{Real, AbstractArray}... that are safe.
# [1] addtoindex!(A::AbstractArray, X, I::Union{Colon, Real, AbstractRange, AbstractArray{Bool,N} where N}...) in AutoGrad at /dev/shm/dyuret/.julia/packages/AutoGrad/VFrAv/src/addto.jl:133
# This specifically catches cases where indices can repeat (i.e. arrays of ints or cartesianindices) and handles them
# [2] addtoindex!(A::AbstractArray, x, I::Union{Real, AbstractArray}...) in AutoGrad at /dev/shm/dyuret/.julia/packages/AutoGrad/VFrAv/src/addto.jl:107
# This is most general and handles cases that are not caught by #2 that are safe.
# [3] addtoindex!(A::AbstractArray, X, I...) in AutoGrad at /dev/shm/dyuret/.julia/packages/AutoGrad/VFrAv/src/addto.jl:135

# This extends #1, do KnetArray and CuArray separately because CuArray <: AbstractArray which may cause ambiguity
addtoindex!(A::KnetArray, X, I::Union{Real,Colon,AbstractRange,AbstractArray{Bool}}...) = setindex!(A, addto!(getindex(A,I...), X), I...)
addtoindex!(A::CuArray, X, I::Union{Real,Colon,AbstractRange,AbstractArray{Bool}}...) = setindex!(A, addto!(getindex(A,I...), X), I...)

# These extend #2, however only a small number of cases have been implemented, the rest should throw method errors
addtoindex!(A::KnetArray, X, I::Union{Real, AbstractArray}...) = _addtoindex!(A, X, I...)
addtoindex!(A::CuArray, X, I::Union{Real, AbstractArray}...) = _addtoindex!(A, X, I...)
addtoindex!(A::KnetArray, X, I::Union{Real, Colon, AbstractArray}...) = _addtoindex!(A, X, I...)
addtoindex!(A::CuArray, X, I::Union{Real, Colon, AbstractArray}...) = _addtoindex!(A, X, I...)
addtoindex!(A::KnetArray, X, I::AbstractArray{<:CartesianIndex}) = _addtoindex!(A, X, c2i(size(A),I))
addtoindex!(A::CuArray, X, I::AbstractArray{<:CartesianIndex}) = _addtoindex!(A, X, c2i(size(A),I))
c2i(d::Dims,i::AbstractArray{<:CartesianIndex}) = Int32[(LinearIndices(d))[c.I...] for c in i]

# This extends #3, which is the remaining cases safe to setindex!
addtoindex!(A::KnetArray, X, I...) = setindex!(A, addto!(getindex(A,I...), X), I...)
addtoindex!(A::CuArray, X, I...) = setindex!(A, addto!(getindex(A,I...), X), I...)

# These are the actual implementations for #2 that can handle repeat indices:
for (R,P) in ((CuArray,CuPtr),(KnetArray,Ptr)), T in (Float16, Float32, Float64); F = sizeof(T)*8
    @eval begin

        function _addtoindex!(A::$R{$T}, X, I::AbstractArray{<:Real})
            I = convert($R{Int32},I)
            X = convert($R{$T},X)
            @knet8($("addents_$F"),(Cint,$P{Int},$P{$T},$P{$T}),
                   length(I), I, A, X)
            return A
        end

        function _addtoindex!(A::$R{$T}, X, ::Colon, I::AbstractArray{<:Real})
            I = convert($R{Int32},I)
            X = convert($R{$T},X)
            @knet8($("addcols_$F"),(Cint,Cint,Cint,$P{Int},$P{$T},$P{$T}),
                   size(A,1), size(A,2), length(I), I, A, X)
            return A
        end

        function _addtoindex!(A::$R{$T}, X, I::AbstractArray{<:Real}, ::Colon)
            I = convert($R{Int32},I)
            X = convert($R{$T},X)
            @knet8($("addrows_$F"),(Cint,Cint,Cint,$P{Int},$P{$T},$P{$T}),
                   size(A,1), size(A,2), length(I), I, A, X)
            return A
        end

    end
end
