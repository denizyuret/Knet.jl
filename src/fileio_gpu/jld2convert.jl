using JLD2
using Knet.KnetArrays: KnetArray
using CUDA: CuArray

struct JLD2KnetArray{T,N}; array::Array{T,N}; end
JLD2.writeas(::Type{KnetArray{T,N}}) where {T,N} = JLD2KnetArray{T,N}
JLD2.wconvert(::Type{JLD2KnetArray{T,N}}, x::KnetArray{T,N}) where {T,N} = JLD2KnetArray(Array(x))
JLD2.rconvert(::Type{KnetArray{T,N}}, x::JLD2KnetArray{T,N}) where {T,N} = KnetArray(x.array)

struct JLD2CuArray{T,N}; array::Array{T,N}; end
JLD2.writeas(::Type{CuArray{T,N}}) where {T,N} = JLD2CuArray{T,N}
JLD2.wconvert(::Type{JLD2CuArray{T,N}}, x::CuArray{T,N}) where {T,N} = JLD2CuArray(Array(x))
JLD2.rconvert(::Type{CuArray{T,N}}, x::JLD2CuArray{T,N}) where {T,N} = CuArray(x.array)

