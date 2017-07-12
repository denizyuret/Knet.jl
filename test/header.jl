if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
    using Combinatorics
else
    using BaseTestNext
    const Test = BaseTestNext
    Base.randn{T}(::Type{T},dims::Integer...)=convert(Array{T},randn(dims...))
    Base.randn{T}(::Type{T},dims::Dims)=convert(Array{T},randn(dims...))
end

using Knet, KArrays
