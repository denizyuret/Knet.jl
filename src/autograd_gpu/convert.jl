import Base: Array, convert
import AutoGrad: back
import CUDA: CuArray
using Knet.KnetArrays: DevArray, KnetArray
using AutoGrad: forw, Value, Arg, value

Array(x::Value{K}) where {K<:DevArray}=convert(Array,x)
KnetArray(x::Value{A}) where {A<:AbstractArray}=convert(KnetArray,x)
CuArray(x::Value{A}) where {A<:AbstractArray}=convert(CuArray,x)

convert(::Type{A},x::Value{K}) where {A<:AbstractArray,K<:DevArray}=forw(convert,A,x)
convert(::Type{K},x::Value{A}) where {A<:AbstractArray,K<:DevArray}=forw(convert,K,x)
back(::typeof(convert),::Type{Arg{2}},dy,y,T,x) = convert(typeof(value(x)),dy)

# Array/KnetArray Transfer

# So we will define gradients for convert, KnetArray, Array manually:

# This works but unnecessarily defines new functions:
# cpu2gpu(x::Array)=KnetArray(x)
# @primitive cpu2gpu(x),dy,y (gpu2cpu(dy))
# gpu2cpu(x::KnetArray)=Array(x)
# @primitive gpu2cpu(x),dy,y (cpu2gpu(dy))

# This does not work because !isa(Array,Function)
# @primitive  KnetArray(x::Array),dy  Array(dy)
# @primitive  Array(x::KnetArray),dy  KnetArray(dy)

# This does not work, parametric methods not yet supported, also unnecessary first arg gradient.
# @primitive convert{A<:AbstractArray,K<:KnetArray}(T::Type{K}, x::Value{A}),dy 0 Array(dy)
# @primitive convert{A<:AbstractArray,K<:KnetArray}(T::Type{A}, x::Value{K}),dy 0 KnetArray(dy)

# This gives ambiguity errors:
# @primitive convert(t::Type,x::KnetArray),dy  nothing  convert(KnetArray,dy)
# @primitive convert(t::Type{KnetArray},x::AbstractArray),dy  nothing  convert(Array,dy)

