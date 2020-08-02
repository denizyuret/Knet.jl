using CUDA
import Base: getindex

## Indexing with Int array: used in nll.

function getindex(x::CuArray{T}, i::AbstractArray{I}) where {T,I<:Real}
    y = similar(x, size(i))
    if isempty(y); return y; end
    i = Array{Int32}(i)
    checkbetween(i, 1, length(x))
    unsafe_getindex!(x,y,CuArray{Int32}(i))
    return y
end

for F in (32,64); T=Symbol("Float$F"); @eval begin

## Indexing with CuArray{Int32}: low level, only Int32 supported, no bounds checking

    function unsafe_getindex!(x::CuArray{$T}, y::CuArray{$T}, i::CuArray{Int32})
        @knet8($("getents_$F"),(Cint,CuPtr{Int},CuPtr{$T},CuPtr{$T}), length(i), i, x, y)
        return y
    end

end; end

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

import Base.convert
convert(::Type{Array{T}}, x::CuArray{S,N}) where {T,N,S} = convert(Array{T,N}, x)
convert(::Type{Array{T,N}}, x::CuArray{S,N}) where {T,N,S} = convert(Array{T,N},Array(x))
