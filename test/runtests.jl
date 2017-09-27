#                            commit 8.3 8.3 6cb 6cb 8.2 6cb 6cb
#                           machine ai5 ai4 tr5 tr4 aws osx os4
#@time include("kptr.jl")         #   1   1   0   0   2   0   0
using GPUArrays
Base.isapprox(a::AbstractArray, b::GPUArray) = isapprox(a, Array(b))
Base.isapprox(a::GPUArray, b::AbstractArray) = isapprox(Array(a), (b))
Base.isapprox(a::GPUArray, b::GPUArray) = mapreduce((a, b)-> Cuint(a ≈ b), +, Cuint(0), a, b) == length(a)

Base.:(==)(a::AbstractArray, b::GPUArray) = ==(a, Array(b))
Base.:(==)(a::GPUArray, b::AbstractArray) = ==(Array(a), (b))
function Base.:(==)(a::GPUArray, b::GPUArray)
    length(a) == length(b) || return false
    mapreduce((a, b)-> Cuint(a == b), +, Cuint(0), a, b) == length(a)
end

@time include("gpu.jl")           #   1   1   0   0  13   0   0  8 pass
@time include("distributions.jl") #   1   1   2   1   2   3   2  3 pass
@time include("update.jl")        #  29  26 100  22 103  25  23  9 pass, 9 error
@time include("karray.jl")        #  19  12   -   -  16   -   0  185 pass, 3 fail, 60 error
@time include("linalg.jl")        #  24  14  22   7  19  33  19  112 pass, 74 fail, 6 error
@time include("broadcast.jl")     #  34  19 491 119  32  53  25  3773 pass, 3 error
@time include("unary.jl")         #  42   6  36   4  44  67  11  3 pass, 1 error ?? 1406
@time include("reduction.jl")     #  21  29  11  32  55  29  1504 pass, 48 failed, 320 errored
# @time include("conv.jl")          #  22  12  62  47  34  44  16  0 pass

gc()
8 + 3 + 9 + 185 + 112 + 3773 + 1405 + 1504
