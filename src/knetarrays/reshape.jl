import Base: reshape, vec, permutedims, permutedims!

function reshape(a::KnetArray{T}, dims::Dims) where T
    if dims==size(a)
        a
    elseif prod(dims) != length(a)
        throw(DimensionMismatch())
    else
        KnetArray{T,length(dims)}(a.ptr, dims)
    end
end

reshape(a::KnetArray, dims::Union{Int,Colon}...) = reshape(a, dims)
reshape(a::KnetArray, dims::Tuple{Vararg{Union{Int,Colon}}}) = reshape(a, Base._reshape_uncolon(a, dims))

vec(a::KnetArray) = reshape(a, length(a))

permutedims!(y::KnetArray, x::KnetArray, perm) = (permutedims!(CuArray(y), CuArray(x), perm); y)

# Based on permutedims, multidimensional.jl:1334, julia 1.2.0
function permutedims(B::KnetArray,perm)
    dimsB = size(B)
    ndimsB = length(dimsB)
    (ndimsB == length(perm) && isperm(perm)) || throw(ArgumentError("no valid permutation of dimensions"))
    dimsP = ntuple(i->dimsB[perm[i]], ndimsB)::typeof(dimsB)
    P = similar(B, dimsP)
    permutedims!(P,B,perm)
end

#permutedims(x::KnetMatrix)=permutedims(x,(2,1))  # CUDA.jl is %10 faster but has startup cost
permutedims(x::KnetMatrix)=_transpose(x)          # cuDNN is %10 slower but no startup cost
permutedims(x::KnetVector)=copy(reshape(x,1,:))

