### Use CuArrays kernels as fallback for undefined KnetArray operations.

using CUDA: CuPtr, unsafe_free!, usage_limit, CURAND
import CUDA: CuArray
import Base: getindex, setindex!, permutedims, permutedims!, cat, hcat, vcat, unsafe_convert
import Random: rand!, randn!

# Extend function CuArray to create a memory shared CuArray from KnetArray:
# Avoid the cu function as it changes eltype to Float32
function CuArray(x::KnetArray{T}) where {T}
    p = CuPtr{T}(UInt(x.ptr.ptr))
    Base.unsafe_wrap(CuArray{T}, p, size(x); own=false)
end

function unsafe_convert(T::Type{<:CuPtr}, x::KnetArray)
    T(UInt(x.ptr.ptr))
end

# Based on julia-1.4.2/base: getindex@abstractarray.jl:980, _getindex@multidimensional.jl:726, _unsafe_getindex!@multidimensional.jl:738
function getindex(A::KnetArray, I...)
    _A = CuArray(A)
    I = Base.to_indices(_A, I)
    checkbounds(_A, I...)
    shape = Base.index_shape(I...)
    B = similar(A, length.(shape))
    _B = CuArray(B)
    Base._unsafe_getindex!(_B, _A, I...)
    return B
end

function setindex!(A::KnetArray, B, I...)
    if B isa KnetArray || B isa AbstractArray
        B = CuArray(B)
    end
    setindex!(CuArray(A), B, I...)
    return A
end

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

using Base: dims2cat, cat_shape, __cat

# vcat(X::KnetArray...)=cat(X...; dims=Val(1)) # karray.jl version is 30%-80% faster
hcat(X::KnetArray...)=cat(X...; dims=Val(2))   # This should only kick in for dims > 2, karray.jl 1/2D versions are 100% faster

# Based on _cat_t, abstractarray.jl:1439, julia 1.2.0
function cat(X::KnetArray{T}...; dims) where {T}
    catdims = dims2cat(dims)
    # catdims == (true,) && return vcat_old(X...) # 10-30% faster
    shape = cat_shape(catdims, (), map(size, X)...)
    # length(shape) <= 2 && catdims == (false,true) && return hcat_old(X...) # 50% faster
    A = similar(X[1], T, shape) # cat_similar(X[1], T, shape)
    if T <: Number && count(!iszero, catdims) > 1
        fill!(A, zero(T))
    end
    __cat(CuArray(A), shape, catdims, map(CuArray,X)...)
    return A
end

# Must be careful with memory management, for now we will let Knet manage memory.
# use CuArray(x) with overwriting kernels only.
# use the following with caution.

if has_cuda()
    function KnetArray(x::CuArray{T,N}) where {T,N}
        p = Base.bitcast(Knet.Cptr, x.ptr)
        k = Knet.KnetPtr(p, sizeof(x), gpu(), x) 
        KnetArray{T,N}(k, size(x))
    end
end

# Testing the CUDA.jl allocator: set Knet.cuallocator()=true to use this
function KnetPtrCu(len::Int)
    c = CuArray{UInt8}(undef, len)
    p = convert(Cptr, convert(UInt, Base.unsafe_convert(CuPtr{UInt8}, Base.cconvert(CuPtr{UInt8}, c))))
    kp = KnetPtr(p, len, gpu(), c)
    finalizer(freeKnetPtrCu, kp)
end

function freeKnetPtrCu(p::KnetPtr)
    # GC.gc comes here directly, manual calls come through freeKnetPtr()
    # After a manual call, GC.gc may call the finalizer again, avoid double free
    if p.parent isa CuArray
        unsafe_free!(p.parent)
        p.ptr, p.parent = C_NULL, nothing
    elseif p.parent isa Nothing
        @assert p.ptr == C_NULL
        # already freed, do nothing
    elseif p.parent isa KnetPtr
        # subarray, do nothing
    else
        error("Bad parent pointer $(typeof(p.parent))")
    end
end

# argmax, argmin etc. Fixes https://github.com/denizyuret/Knet.jl/issues/368.
# Two options: argmax(Array(KnetArray)) vs argmax(CuArray)
# Experiments: 10x10, 100x100, 1000x1000 with no dims, dims=1, dims=2
# With no dims, CUDA.jl is better for 100x100, 1000x1000.
# With all others, KnetArray is better.

import Base: argmax, argmin, findmax, findmin
# TODO: try this again after https://github.com/JuliaGPU/CuArrays.jl/issues/304 is resolved
# argmaxarray(x,d)=((d===:) && length(x) > 4096 ? CuArray(x) : Array(x))
argmaxarray(x,d)=Array(x)
argmax(x::KnetArray; dims=:)=argmax(argmaxarray(x,dims); dims=dims)
argmin(x::KnetArray; dims=:)=argmin(argmaxarray(x,dims); dims=dims)
findmax(x::KnetArray; dims=:)=findmax(argmaxarray(x,dims); dims=dims)
findmin(x::KnetArray; dims=:)=findmin(argmaxarray(x,dims); dims=dims)


# Issue #108:Element-wise power of KnetArray give NaN results #108
# This is a bug with CUDA giving NaN for integer powers of negative numbers (powf is broken)

import Base.Broadcast: broadcasted

function broadcasted(::typeof(^),a::KnetArray{T},s::Number) where T
    b = similar(a)
    ca = CuArray(a)
    cb = CuArray(b)
    cb .= ca .^ T(s)
    return b
end

function broadcasted(::typeof(^),s::Number,a::KnetArray{T}) where T
    b = similar(a)
    ca = CuArray(a)
    cb = CuArray(b)
    cb .= T(s) .^ ca
    return b
end

# Functions from old random.jl:

rand!(a::KnetArray)=(rand!(CuArray(a)); a)
randn!(a::KnetArray)=(randn!(CuArray(a)); a)
