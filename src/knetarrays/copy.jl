import Base: copyto!, copy, deepcopy_internal
# include("karray.jl") ## _unsafe_copy!, KnetArray

# Generalizing low level copy using linear indexing to/from gpu arrays:
# copyto!{T}(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer)
# Note that this is an unsafe operation, no argument or bounds checking performed.
# Defined in Base:
# _unsafe_copy!{T}(dest::Ptr{T}, src::Ptr{T}, n) at array.jl:73
# _unsafe_copy!{T}(dest::Array{T,N}, doffs, src::Array{T,N}, soffs, n) at array.jl:79

const KorA{T} = Union{KnetArray{T},Array{T}}

function _copyto!(dest::KorA{T}, doffs::Integer, src::KorA{T}, soffs::Integer, n::Integer) where {T}
    if n == 0; return dest; end
    if n < 0; throw(ArgumentError()); end
    if soffs < 1 || doffs < 1 || soffs+n-1 > length(src) || doffs+n-1 > length(dest)
        throw(BoundsError())
    end
    _unsafe_copy!(dest, doffs, src, soffs, n)
end

function _copyto!(dest::KorA{T}, src::KorA{T}) where {T}
    if length(dest) < length(src); throw(BoundsError()); end
    copyto!(dest, 1, src, 1, length(src))
end

# Avoid Array->Array to prevent base conflict.
copyto!(dest::KnetArray{T}, doffs::Integer, src::KnetArray{T}, soffs::Integer, n::Integer) where {T} = _copyto!(dest, doffs, src, soffs, n)
copyto!(dest::KnetArray{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer) where {T} = _copyto!(dest, doffs, src, soffs, n)
copyto!(dest::Array{T}, doffs::Integer, src::KnetArray{T}, soffs::Integer, n::Integer) where {T} = _copyto!(dest, doffs, src, soffs, n)
copyto!(dest::KnetArray{T}, src::KnetArray{T}) where {T} = _copyto!(dest, src)
copyto!(dest::KnetArray{T}, src::Array{T}) where {T} = _copyto!(dest, src)
copyto!(dest::Array{T}, src::KnetArray{T}) where {T} = _copyto!(dest, src)


function copy(a::KnetArray)
    _unsafe_copy!(similar(a),1,a,1,length(a))
end

# This will make deepcopy work properly
deepcopy_internal(x::KnetArray, s::IdDict)=if haskey(s,x); s[x]; else; copy(x); end

