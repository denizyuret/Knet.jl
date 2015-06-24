# This file contains various utilities, compatibility fixes and hacks.
# Hopefully it will shrink down to nothing as things get fixed in the
# original packages.

import Base: copy, copy!, rand!, fill!, convert, reshape, isempty, similar, full, sparse
# these probably should go into type based files for cpu/gpu dense/sparse.
import Base: Ac_mul_B, A_mul_Bc, Ac_mul_Bc
import Base: A_mul_Bt,  At_mul_B
import Base: A_mul_Bt!, At_mul_B!, A_mul_B!


# Print date, expression and elapsed time after execution
VERSION < v"0.4-" && eval(Expr(:using,:Dates))
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end

issimilar(a,b)=((typeof(a)==typeof(b)) && (size(a)==size(b)))
size2(y)=(nd=ndims(y); (nd==1 ? (length(y),1) : (stride(y, nd), size(y, nd))))
accuracy(y,z)=mean(findmax(y,1)[2] .== findmax(z,1)[2])
isongpu(a)=(GPU && isa(a, AbstractCudaArray))
itype{Tv,Ti}(::SparseMatrixCSC{Tv,Ti})=Ti
similar{Tv,Ti}(::SparseMatrixCSC{Tv,Ti},m,n)=spzeros(Tv,Ti,m,n) # this is missing

similar!(l, n, a, T, dims::Integer...; o...) = similar!(l,n,a,T,dims; o...)
similar!(l, n, a, dims::Integer...; o...) = similar!(l,n,a,dims; o...)
similar!(l, n, a, dims::Dims; o...) = similar!(l,n,a,eltype(a),dims; o...)

function similar!(l, n, a, T=eltype(a), dims=size(a); fill=nothing)
    if !isdefined(l,n) || (size(l.(n)) != dims)
        if isa(a, AbstractSparseArray)
            l.(n) = spzeros(T, itype(a), dims...)
            fill != nothing && fill != 0 && error("Cannot fill sparse with $fill")
        else
            l.(n) = similar(a, T, dims)
            fill != nothing && fill!(l.(n), fill)
        end
    end
    return l.(n)
end

if GPU   ########## CUDA extensions:

const libkunet = find_library(["libkunet"], [Pkg.dir("KUnet/src")])

convert{T,S}(::Type{CudaArray{T}}, x::Array{S})=CudaArray(convert(Array{T}, x))
convert{T,S}(::Type{Array{T}}, x::CudaArray{S})=convert(Array{T}, to_host(x))
reshape(a::CudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
reshape(a::CudaArray, dims::Int...)=reshape(a, dims)
rand!(A::CudaArray{Float32})=(ccall((:randfill32,libkunet),Void,(Cint,Ptr{Float32}),length(A),A); A)
rand!(A::CudaArray{Float64})=(ccall((:randfill64,libkunet),Void,(Cint,Ptr{Float64}),length(A),A); A)
fill!(A::CudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)
isempty(A::CudaArray)=(length(A)==0)

typealias CopyableArray{T} Union(Array{T},SubArray{T},HostArray{T},CudaArray{T})

function copy!{T}(dst::CopyableArray{T}, di::Integer, src::CopyableArray{T}, si::Integer, n::Integer; stream=null_stream)
    if si+n-1 > length(src) || di+n-1 > length(dst) || di < 1 || si < 1
        throw(BoundsError())
    end
    nbytes = n * sizeof(T)
    dptr = pointer(dst) + (di-1) * sizeof(T)
    sptr = pointer(src) + (si-1) * sizeof(T)
    CUDArt.rt.cudaMemcpyAsync(dptr, sptr, nbytes, CUDArt.cudamemcpykind(dst, src), stream)
    return dst
end

# For debugging
function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end

# our version of srand sets both gpu and cpu
function gpuseed(n::Integer)
    srand(n)
    GPU && ccall((:gpuseed,libkunet),Void,(Culonglong,),convert(Culonglong, n))
end

# matmul.jl: Linear algebra extended to CudaArrays (this is partial, todo in cublas)


Ac_mul_B{T<:Real}(A::CudaMatrix{T}, B::CudaMatrix{T}) = At_mul_B(A, B)
At_mul_B{T<:Real}(A::CudaMatrix{T}, B::CudaMatrix{T}) = At_mul_B!(similar(B,(size(A,2),size(B,2))),A, B)
At_mul_B!{T<:Real}(C::CudaMatrix{T}, A::CudaMatrix{T}, B::CudaMatrix{T}) = gemm!('T','N',one(T),A,B,zero(T),C)
full(A::CudaArray)=A

# without this patch, deepcopy does not work on structs with CudaArrays
function Base.deepcopy_internal(x::CudaArray, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    copy(x)
end


cpucopy(x::CudaArray)=to_host(x)
cpucopy(x::AbstractArray)=(isbits(eltype(x)) ? copy(x) : map(cpucopy, x))
cpucopy(x)=mydeepcopy(x, cpucopy)
gpucopy(x::CudaArray)=copy(x)
gpucopy(x::AbstractArray)=(isbits(eltype(x)) ? CudaArray(x) : map(gpucopy, x))
gpucopy(x)=mydeepcopy(x, gpucopy)

# Adapted from deepcopy.jl:29 _deepcopy_t()
function mydeepcopy(x,fn)
    T = typeof(x)
    if T.names===() || !T.mutable || (T==Function)
        return x
    end
    ret = ccall(:jl_new_struct_uninit, Any, (Any,), T)
    for i in 1:length(T.names)
        if isdefined(x,i)
            ret.(i) = fn(x.(i))
        end
    end
    return ret
end

# cpu/gpu dense/sparse matrix ops

# cpu/sparse

At_mul_B!(k::Matrix, x::SparseMatrixCSC, s::SparseMatrixCSC)=A_mul_B!(k,x',s)

function A_mul_B!(k::Matrix, x::SparseMatrixCSC, s::SparseMatrixCSC) # 1607
    @assert size(k)==(size(x,1), size(s,2))
    fill!(k, zero(eltype(k)))
    @inbounds @simd for scol=1:size(s,2)
        @inbounds @simd for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
            srow = s.rowval[sp]
            sval = s.nzval[sp]  # 133
            @inbounds @simd for xp=x.colptr[srow]:(x.colptr[srow+1]-1)
                xrow = x.rowval[xp] # 63
                xval = x.nzval[xp]  # 217
                yinc = xval * sval  # 245
                k[xrow,scol] += yinc # 789
            end
        end
    end
    return k
end

# gpu/sparse

# gpu/dense
A_mul_Bt!{T}(k::CudaMatrix{T}, x::CudaMatrix{T}, s::CudaMatrix{T})=gemm!('N','T',one(T),x,s,zero(T),k)
At_mul_B!{T}(k::CudaMatrix{T}, x::CudaMatrix{T}, s::CudaMatrix{T})=gemm!('T','N',one(T),x,s,zero(T),k)

# cpu/gpu sparse/dense hcat! needed by kperceptron

# hcat!(a,b,vj,nj)=[a b[:,vj[1:nj]]]

hcat!{Tv,Ti<:Integer}(a::Matrix{Tv}, b::Matrix{Tv}, vj::Vector{Ti}, nj::Integer)=[a b[:,vj[1:nj]]]

function hcat!{Tv,Ti<:Integer}(a::SparseMatrixCSC{Tv}, b::SparseMatrixCSC{Tv}, vj::Vector{Ti}, nj::Integer)
    # a: m, n, colptr, rowval, nzval
    # colptr[i]: starting index (in rowval,nzval) of column i
    # colptr[n+1]: nz+1
    @assert size(a,1) == size(b,1)
    @inbounds for i=1:nj
        j = vj[i]  # concat b[:,j]
        b0 = b.colptr[j]
        b1 = b.colptr[j+1]-1
        nz = b1-b0+1
        a.colptr = push!(a.colptr, a.colptr[end]+nz)
        if nz > 0
            a.rowval = append!(a.rowval, b.rowval[b0:b1])
            a.nzval = append!(a.nzval, b.nzval[b0:b1])
        end
    end
    a.n += nj
    return a
end


function hcat!(a::CudaMatrix, b::Union(CudaMatrix,Matrix), vj::Vector, nj::Integer)
    @assert size(a,1) == size(b,1)
    @assert eltype(a) == eltype(b)
    (nrows,ncols) = size(a)
    c = CudaArray(eltype(a), nrows, ncols+nj)   # TODO: is there realloc?
    copy!(c, 1, a, 1, length(a))
    nc = length(a)+1
    for i=1:nj
        nb = (vj[i]-1)*nrows+1
        copy!(c, nc, b, nb, nrows)
        nc += nrows
    end
    return c
end


else  # if GPU

# Need this so code works without gpu
cpucopy(x)=deepcopy(x)
gpucopy(x)=error("No GPU")

end   # if GPU

