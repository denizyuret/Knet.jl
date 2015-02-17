using CUDArt
using CUBLAS
typealias Cmat Ptr{Float32}
const libjnet = find_library(["libjnet"], ["."])

import Base: similar
similar{T}(a::CudaArray{T}, dims::Int...) = similar(a, T, dims)  # abstractarray.jl:134

import Base: pointer
pointer{T}(g::CudaArray{T}, i::Integer) = g.ptr + (i-1) * sizeof(T)

import Base: copy!
import CUDArt: ContiguousArray, rt, cudamemcpykind, cudacopy!
function cudacopy!{T}(dst::ContiguousArray{T}, dstI::Integer, src::ContiguousArray{T}, srcI::Integer, nelem::Integer; stream=null_stream)
    if (dstI < 1 || dstI + nelem - 1 > length(dst) || srcI < 1 || srcI + nelem - 1 > length(src))
        throw(ArgumentError("Bad copy offset/length."))
    end
    nbytes = nelem * sizeof(T)
    rt.cudaMemcpyAsync(pointer(dst,dstI), pointer(src,srcI), nbytes, cudamemcpykind(dst, src), stream)
    return dst
end
copy!(d::CudaArray,di::Integer,s::Array,si::Integer,n::Integer)=cudacopy!(d,di,s,si,n)
copy!(d::Array,di::Integer,s::CudaArray,si::Integer,n::Integer)=cudacopy!(d,di,s,si,n)

import InplaceOps: mul!, badd!
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::CudaVecOrMat) = CUBLAS.gemm!('N','N',one(eltype(A)),A,B,zero(eltype(A)),O)  # InplaceOps.jl:53
badd!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaVecOrMat) = ccall((:badd,libjnet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,B) # InplaceOps.jl:83

reluforw(l,y::CudaArray)=ccall((:reluforw,libjnet),Void,(Cint,Cmat),length(y),y)

function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end

