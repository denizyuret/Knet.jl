import Base: copyto!, copy, deepcopy_internal
using CUDA: CuArray
# include("karray.jl") ## KnetArray

# Avoid Array->Array to prevent base conflict.
copyto!(dest::KnetArray{T}, doffs::Integer, src::KnetArray{T}, soffs::Integer, n::Integer) where {T} = (copyto!(CuArray(dest), doffs, CuArray(src), soffs, n); dest)
copyto!(dest::KnetArray{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer) where {T} = (copyto!(CuArray(dest), doffs, src, soffs, n); dest)
copyto!(dest::Array{T}, doffs::Integer, src::KnetArray{T}, soffs::Integer, n::Integer) where {T} = (copyto!(dest, doffs, CuArray(src), soffs, n); dest)
copyto!(dest::KnetArray{T}, src::KnetArray{T}) where {T} = copyto!(dest, 1, src, 1, length(dest))
copyto!(dest::KnetArray{T}, src::Array{T}) where {T} = copyto!(dest, 1, src, 1, length(dest))
copyto!(dest::Array{T}, src::KnetArray{T}) where {T} = copyto!(dest, 1, src, 1, length(dest))

# Changing types
copyto!(dest::KnetArray{T}, doffs::Integer, src::KnetArray{S}, soffs::Integer, n::Integer) where {T,S} = copyto!(dest, doffs, convert(KnetArray{T},src), soffs, n)
copyto!(dest::KnetArray{T}, doffs::Integer, src::Array{S}, soffs::Integer, n::Integer) where {T,S} = copyto!(dest, doffs, convert(Array{T},src), soffs, n)
copyto!(dest::Array{T}, doffs::Integer, src::KnetArray{S}, soffs::Integer, n::Integer) where {T,S} = copyto!(dest, doffs, convert(KnetArray{T},src), soffs, n)
copyto!(dest::KnetArray{T}, src::KnetArray{S}) where {T,S} = copyto!(dest, convert(KnetArray{T},src))
copyto!(dest::KnetArray{T}, src::Array{S}) where {T,S} = copyto!(dest, convert(Array{T},src))
copyto!(dest::Array{T}, src::KnetArray{S}) where {T,S} = copyto!(dest, convert(KnetArray{T},src))

# This will make deepcopy work properly
copy(a::KnetArray) = copyto!(similar(a), a)
deepcopy_internal(x::KnetArray, s::IdDict) = (haskey(s,x) ? s[x] : copy(x))
