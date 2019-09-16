# This is from https://github.com/JuliaGPU/CUDAapi.jl/pull/84/files

if find_cuda_library("cuda", tk) != nothing # has_cuda()
    try
        import CuArrays: CuArray, CuPtr
    catch ex
        @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
    end
end

### Use CuArrays kernels as fallback for undefined KnetArray operations.

import Base: getindex, setindex!, permutedims, permutedims!, cat, hcat, vcat

# Extend function CuArray to create a memory shared CuArray from KnetArray:
# Avoid the cu function as it changes eltype to Float32
function CuArray(x::KnetArray{T}) where {T}
    p = CuPtr{T}(UInt(x.ptr.ptr))
    Base.unsafe_wrap(CuArray{T}, p, size(x); own=false)
end

# Based on _unsafe_getindex, multidimensional.jl:679, julia 1.2.0
function getindex(A::KnetArray, I...)
    I = Base.to_indices(A, I)
    shape = Base.index_shape(I...)
    B = similar(A, length.(shape))
    Base._unsafe_getindex!(CuArray(B), CuArray(A), I...)
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

#permutedims(x::KnetMatrix)=permutedims(x,(2,1))  # CuArrays is %10 faster but has startup cost
permutedims(x::KnetMatrix)=_transpose(x)          # cuDNN is %10 slower but no startup cost
permutedims(x::KnetVector)=copy(reshape(x,1,:))

# TODO: delete this when fixed in AutoGrad (after 1.1.4):
if first(methods(permutedims, (AutoGrad.Value,))).nargs == 3 # old AutoGrad has permutedims(x,d...)
    @primitive permutedims(x),dy  reshape(permutedims(dy),size(x)) # need reshape for vectors
end

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
# Do not extend function ka to create a memory shared KnetArray from CuArray:
# best not to use CuArrays memory manager simultaneously with KnetArrays memory manager.
# use CuArray(x) with overwriting kernels only.

# function Knet.ka(x::CuArray{T,N}) where {T,N}
#     p = Base.bitcast(Knet.Cptr, x.buf.ptr)
#     k = Knet.KnetPtr(p, sizeof(x), gpu(), x) 
#     # finalizer(identity, k) # hacky way to avoid gc? gives error in running finalizer
#     KnetArray{T,N}(k, size(x))
# end


# Testing the CuArrays allocator: set Knet.cuallocator()=true to use this
function KnetPtrCu(len::Int)
    c = CuArray{UInt8}(undef, len)
    p = convert(Cptr, convert(Int, c.buf.ptr))
    KnetPtr(p, len, gpu(), c)
end
