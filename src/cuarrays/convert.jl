import Base.convert
using CUDA: CuArray

convert(::Type{Array{T}}, x::CuArray{S,N}) where {T,N,S} = convert(Array{T,N}, x)
convert(::Type{Array{T,N}}, x::CuArray{S,N}) where {T,N,S} = convert(Array{T,N},Array(x))
