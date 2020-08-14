import Base: cat, hcat, vcat
import AutoGrad: back
using AutoGrad: Arg, Value, forw, uncat
using Base: dims2cat, cat_shape, __cat
using CUDA: CuArray

# Concatenation:

# Need to extend cat definitions from AutoGrad/src/cat.jl:
const NAVK = Union{Number,AbstractArray,Value,KnetArray}
cat(X::NAVK...; dims) = forw(cat,X...;dims=dims)
back(::typeof(cat),::Type{Arg{N}},y1::NAVK,y::NAVK,x::NAVK...; dims) where {N}=uncat(y1,N,dims,x...)

# Benchmarks in μs for hcat and vcat: a=rand(1000,1000) v=rand(1000), t=v'
#		cpu	gpu	g->c->g	vkernel
# hcat(a,a)	2350	225	16160
# hcat(a,v)	1230	115	6490
# hcat(v,a)	1220	120	6490
# hcat(v,v)	3.53	12.53	48.49
# vcat(a,a)	2630	10980	16590	665
# vcat(a,t)	1350	10860	6550	338
# vcat(t,a)	1360	10850	6570	338
# vcat(v,v)	2.13	12.33	45.40	13.58

# setindex! methods called by hcat/vcat:
# hcat(v,v): I = (Colon(),1:1) I = (Colon(),2:2)
# vcat(v,v): uses single index
# hcat(m,m): I = (Colon(),1:5) I = (Colon(),6:10)
# vcat(m,m): I = (1:3,Colon()) I = (4:6,Colon())

# based on _typed_hcat{T}(::Type{T}, A::AbstractVecOrMat...) in base/abstractarray.jl:1316 julia-1.2.0
function hcat(A::KnetVecOrMat{T}...) where {T}
    nargs = length(A)
    nrows = size(A[1], 1)
    ncols = 0
    for j = 1:nargs
        Aj = A[j]
        if size(Aj, 1) != nrows
            throw(ArgumentError("number of rows of each array must match (got $(map(x->size(x,1), A)))"))
        end
        nd = ndims(Aj)
        ncols += (nd==2 ? size(Aj,2) : 1)
    end
    B = similar(A[1], nrows, ncols)
    pos = 1
    for k = 1:nargs
        Ak = A[k]
        n = length(Ak)
        copyto!(B, pos, Ak, 1, n)
        pos += n
    end
    return B
end

function vcat(A::KnetVector{T}...) where {T}
    nargs = length(A)
    nrows = 0
    for a in A
        nrows += length(a)
    end
    B = similar(A[1], nrows)
    pos = 1
    for k = 1:nargs
        Ak = A[k]
        n = length(Ak)
        copyto!(B, pos, Ak, 1, n)
        pos += n
    end
    return B
end

# based on _typed_vcat{T}(::Type{T}, A::AbstractVecOrTuple{AbstractVecOrMat}) in base/abstractarray.jl:1353 julia-1.2.0
function vcat(A::KnetVecOrMat{T}...) where {T}
    nargs = length(A)
    nrows = sum(a->size(a, 1), A)::Int
    ncols = size(A[1], 2)
    for j = 2:nargs
        if size(A[j], 2) != ncols
            throw(ArgumentError("number of columns of each array must match (got $(map(x->size(x,2), A)))"))
        end
    end
    B = similar(A[1], T, nrows, ncols)
    pos = 1
    for k = 1:nargs
        Ak = A[k]
        p1 = pos+size(Ak,1)-1
        B[pos:p1, :] = Ak
        pos = p1+1
    end
    return B
end

function vcat(A::KnetArray{T}...) where {T}
    nargs = length(A)
    size2 = size(A[1])[2:end]
    for j = 2:nargs
        if size(A[j])[2:end] != size2
            throw(ArgumentError("size(a)[2:end] of each array must match (got $(map(size, A)))"))
        end
    end
    A2 = (reshape(a, size(a,1), :) for a in A)
    B = vcat(A2...)
    reshape(B, size(B,1), size2...)
end

function cat_old(a1::KnetVecOrMat{T}, a::KnetVecOrMat{T}...; dims) where {T}
    if     dims==1 || dims==Val(1); vcat(a1, a...)
    elseif dims==2 || dims==Val(2); hcat(a1, a...)
    else error("cat(a...;dims=$dims) not implemented.")
    end
end

# Avoid using Base for unimplemented cat methods:

# using AutoGrad: NA # Union{Number,AbstractArray}
# const NAK = Union{Number,AbstractArray,KnetArray}
# cat(a::NA, as::NA...; dims)=Base._cat(dims, a, as...)
# cat(a::NAK, as::NAK...; dims)=throw(MethodError(cat, (a, as...)))
# hcat(a::NA, as::NA...)=cat(a,as...; dims=2)
# hcat(a::NAK, as::NAK...)=throw(MethodError(hcat, (a, as...)))
# vcat(a::NA, as::NA...)=cat(a,as...; dims=1)
# vcat(a::NAK, as::NAK...)=throw(MethodError(vcat, (a, as...)))

# # Ambiguity fix for abstractarray.jl:1066-1072
# using Base: hvcat_fill, promote_typeof
# vcat(X::Number, Xs::Number...) = hvcat_fill(Array{promote_typeof(X, Xs...)}(undef,1+length(Xs)), (X, Xs...))
# hcat(X::Number, Xs::Number...) = hvcat_fill(Array{promote_typeof(X, Xs...)}(undef,1,1+length(Xs)), (X, Xs...))

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

