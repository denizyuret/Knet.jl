# This does not work yet: https://github.com/JuliaIO/JLD2.jl/issues/40

using JLD2
using Knet.KnetArrays: KnetArray
using CUDA: CuArray

struct JLD2KnetArray; array::Array; end
JLD2.writeas(::Type{KnetArray}) = JLD2KnetArray
JLD2.wconvert(::Type{JLD2KnetArray}, x::KnetArray) = JLD2KnetArray(Array(x))
JLD2.rconvert(::Type{KnetArray}, x::JLD2KnetArray) = KnetArray(x.array)

struct JLD2CuArray; array::Array; end
JLD2.writeas(::Type{CuArray}) = JLD2CuArray
JLD2.wconvert(::Type{JLD2CuArray}, x::CuArray) = JLD2CuArray(Array(x))
JLD2.rconvert(::Type{CuArray}, x::JLD2CuArray) = CuArray(x.array)

