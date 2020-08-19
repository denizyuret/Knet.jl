import Base: ==, isapprox
using LinearAlgebra: norm

# Comparison
(==)(a::KnetArray{T},b::KnetArray{T}) where {T}=(size(a)==size(b) && sum(abs2,a-b)==0)
(==)(a::AbstractArray,b::KnetArray)=(size(a)==size(b) && a==Array(b))
(==)(a::KnetArray,b::AbstractArray)=(size(a)==size(b) && Array(a)==b)
# Adapted from base/linalg/generic.jl:589
isapprox(a::AbstractArray,b::KnetArray;o...)=(size(a)==size(b) && isapprox(a,Array(b);o...))
isapprox(a::KnetArray,b::AbstractArray;o...)=(size(a)==size(b) && isapprox(Array(a),b;o...))

function isapprox(a::KnetArray{T}, b::KnetArray{T}; rtol=sqrt(eps(T)), atol=T(0)) where {T}
    (size(a)==size(b) && norm(a-b) <= atol + rtol * max(norm(a), norm(b)))
end
