issimilar(a,b)=((typeof(a)==typeof(b)) && (size(a)==size(b)))

# Fix bug with deepcopy, where a shared bits array is copied multiple times:
# TODO: check if this bug is still there
Base.deepcopy_internal{T<:Number}(x::Array{T}, s::ObjectIdDict)=(haskey(s,x)||(s[x]=copy(x));s[x])

function Base.isapprox(x, y; 
                       maxeps::Real = max(eps(eltype(x)), eps(eltype(y))),
                       rtol::Real=maxeps^(1/3), atol::Real=maxeps^(1/2))
    size(x) == size(y) || (warn("isapprox: $(size(x))!=$(size(y))"); return false)
    x = convert(Array, x)
    y = convert(Array, y)
    d = abs(x-y)
    s = abs(x)+abs(y)
    maximum(d - rtol * s) <= atol
end

# This is missing from Base
Base.convert{T,I}(::Type{Array{T,2}}, a::SparseMatrixCSC{T,I})=full(a)

# alternatives defined in cudart.jl
if !GPU
    typealias BaseArray{T,N} Union{Array{T,N},SubArray{T,N}}
end

copysync!(a::AbstractArray,b::AbstractArray)=copy!(a,b)
fillsync!(a::AbstractArray,x)=fill!(a,x)

# Define a more versatile version of randn!

import Base: randn!, GLOBAL_RNG

randn!{T}(A::AbstractArray{T}, mean=zero(T), std=one(T)) = randn!(GLOBAL_RNG, A, mean, std)

function randn!{T}(rng::AbstractRNG, A::AbstractArray{T}, mean=zero(T), std=one(T))
    for i in eachindex(A)
        @inbounds A[i] = mean+std*randn(rng)
    end
    A
end

# This is missing from sparse/linalg.jl: modified from (*) line 100.
import Base: A_mul_B!, A_mul_Bt!, At_mul_B!
using Base.LinAlg.BLAS: gemm!

### Add the ability to multiply arrays with other than 2 dimensions
mat2d(x)=(ndims(x)==2 ? x : (x2=reshape(x, size2(x));pointer(x2)===pointer(x)||error();x2))
A_mul_B!{T}(C::Array{T}, A::Array{T}, B::Array{T})=(gemm!('N','N',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); C)
A_mul_Bt!{T}(C::Array{T}, A::Array{T}, B::Array{T})=(gemm!('N','T',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); C)
At_mul_B!{T}(C::Array{T}, A::Array{T}, B::Array{T})=(gemm!('T','N',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); C)

# y = w * x
function Base.A_mul_B!{TX,TvA,TiA}(Y::StridedMatrix{TX}, X::StridedMatrix{TX}, A::SparseMatrixCSC{TvA,TiA})
    mX, nX = size(X)
    nX == A.m || throw(DimensionMismatch())
    size(Y) == (mX, A.n) || throw(DimensionMismatch())
    fill!(Y,0)
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for multivec_row=1:mX, col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        Y[multivec_row, col] += X[multivec_row, rowval[k]] * nzval[k]
    end
    Y
end

# dw = dy * x'
function Base.A_mul_Bt!{TX,TvA,TiA}(Y::SparseMatrixCSC{TvA,TiA}, X::StridedMatrix{TX}, A::SparseMatrixCSC{TvA,TiA})
    error(:cpu_sparse_not_implemented_yet)
end

# dx = w' * dy is all dense


### DEAD CODE:

# # SIMILAR! create an array l.(n) similar to a given one.  If l.(n)
# # exists check and resize if necessary.

# function similar!(l, n, a, T=eltype(a), dims=size(a); fill=nothing)
#     if !isdefined(l,n) || (typeof(l.(n)) != typeof(a))
#         l.(n) = similar(a, T, dims)
#         fill != nothing && fillsync!(l.(n), fill)
#     elseif (size(l.(n)) != dims)
#         l.(n) = resize!(l.(n), dims)
#         fill != nothing && fillsync!(l.(n), fill)
#     end
#     return l.(n)
# end

# similar!(l, n, a, T, dims::Integer...; o...) = similar!(l,n,a,T,dims; o...)
# similar!(l, n, a, dims::Integer...; o...) = similar!(l,n,a,dims; o...)
# similar!(l, n, a, dims::Dims; o...) = similar!(l,n,a,eltype(a),dims; o...)

# issimilar1(a,b)=((eltype(a)==eltype(b)) && (isongpu(a)==isongpu(b)) && (length(a)==length(b)))
# issimilar2(a,b)=((eltype(a)==eltype(b)) && (isongpu(a)==isongpu(b)) && (size2(a)==size2(b)))


# This does not work in place!
# Base.resize!(a::Array, d::Dims)=similar(a, d)

