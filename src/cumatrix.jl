import Base: convert, reshape, rand!, fill!, isempty, full, copy!
import Base: Ac_mul_B, A_mul_Bc, Ac_mul_Bc
import Base: A_mul_Bt,  At_mul_B
import Base: A_mul_Bt!, At_mul_B!, A_mul_B!

convert{T,S}(::Type{CudaArray{T}}, x::Array{S})=CudaArray(convert(Array{T}, x))
convert{T,S}(::Type{Array{T}}, x::CudaArray{S})=convert(Array{T}, to_host(x))
reshape(a::CudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
reshape(a::CudaArray, dims::Int...)=reshape(a, dims)
rand!(A::CudaArray{Float32})=(ccall((:randfill32,libkunet),Void,(Cint,Ptr{Float32}),length(A),A); A)
rand!(A::CudaArray{Float64})=(ccall((:randfill64,libkunet),Void,(Cint,Ptr{Float64}),length(A),A); A)
fill!(A::CudaArray,x::Number)=(isempty(A)||cudnnSetTensor(A, x);A)
isempty(A::CudaArray)=(length(A)==0)
full(A::CudaArray)=A            # this is missing

typealias CopyableArray{T} Union(Array{T},SubArray{T},HostArray{T},CudaArray{T}) # no sparse

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

# matmul.jl: Linear algebra extended to CudaArrays (this is partial, todo in cublas)
Ac_mul_B{T<:Real}(A::CudaMatrix{T}, B::CudaMatrix{T}) = At_mul_B(A, B)
At_mul_B{T}(A::CudaMatrix{T}, B::CudaMatrix{T}) = At_mul_B!(similar(B,(size(A,2),size(B,2))),A, B)
A_mul_Bt!{T}(k::CudaMatrix{T}, x::CudaMatrix{T}, s::CudaMatrix{T})=gemm!('N','T',one(T),x,s,zero(T),k)
At_mul_B!{T}(k::CudaMatrix{T}, x::CudaMatrix{T}, s::CudaMatrix{T})=gemm!('T','N',one(T),x,s,zero(T),k)

# without this patch, deepcopy does not work on structs with CudaArrays
function Base.deepcopy_internal(x::CudaArray, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    copy(x)
end

function hcat!{T}(a::CudaMatrix{T}, b::Union(CudaMatrix{T},Matrix{T}), vj::Vector, nj::Integer)
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
