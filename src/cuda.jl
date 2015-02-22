# This file contains various compatibility fixes and hacks.  Hopefully
# it will shrink down to nothing as things get fixed in the original
# packages.

# This fixes an InplaceOps bug:
import Base: ctranspose
import InplaceOps: Transpose, mul!
ctranspose(x::Matrix)=Transpose(x)  # This was overwritten in base
mul!(O::Matrix, A::Matrix, B::Transpose) = A_mul_Bt!(O,A,B.obj)   # 3rd arg B gives type error
mul!(O::Matrix, A::Transpose, B::Matrix) = At_mul_B!(O,A.obj,B)   # 2nd arg A gives type error

# arrays.jl:297, need this so generic code works with cpu arrays
import Base: copy!
copy!{T}(dst::DenseArray{T}, dstI::(Union(Int,Range1{Int})...), src::DenseArray{T}, srcI::(Union(Int,Range1{Int})...))=copy!(sub(dst, dstI...), sub(src, srcI...))


if isdefined(:CUDArt)   ########## CUDA extensions:

typealias Cmat Ptr{Float32}
const libkunet = find_library(["libkunet"], ["."])

# TODO: these don't hang high enough in the type hierarchy
import InplaceOps: badd!, bmul!, bsub! # TODO: non of these implementations are complete
ctranspose(x::CudaVecOrMat)=Transpose(x)
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::CudaVecOrMat) = CUBLAS.gemm!('N','N',one(eltype(O)),A,B,zero(eltype(O)),O)  # InplaceOps.jl:53
mul!(O::CudaVecOrMat, A::Transpose, B::CudaVecOrMat) = CUBLAS.gemm!('T','N',one(eltype(O)),A.obj,B,zero(eltype(O)),O)
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::Transpose) = CUBLAS.gemm!('N','T',one(eltype(O)),A,B.obj,zero(eltype(O)),O)
badd!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaVecOrMat) = ccall((:badd,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,B) # InplaceOps.jl:83
bmul!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, x::Float32) = CUBLAS.scal!(length(A), x, A, 1)
bsub!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaMatrix) = CUBLAS.axpy!(length(A), -1.0f0, B, 1, A, 1)
bsub!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, x::Float32) = CUBLAS.axpy!(length(A), -1.0f0, CudaArray([x]), 0, A, 1)

# # I could not get this to work:
# import Base: convert, promote_rule
# convert(::Type{Mat},x::Transpose{Mat})=x.obj
# promote_rule(::Type{Mat},::Type{Transpose{Mat}})=Mat

import Base: sum!, zeros, rand!, fill!  
# TODO: add error checking here since this is not a full implementation of sum!
sum!(r::CudaVecOrMat, A::CudaMatrix) = ccall((:bsum,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,r) # reducedim.jl:226
zeros(A::CudaArray)=CUBLAS.scal!(length(A), zero(eltype(A)), copy(A), 1)
rand!(A::CudaArray)=(ccall((:randfill,libkunet),Void,(Cint,Cmat),length(A),A); A)
fill!(A::CudaArray,x::Float32)=(ccall((:fill,libkunet),Void,(Cint,Cfloat,Cmat),length(A),x,A); A)

# TODO: This does not seem to work:
gpuseed(n::UInt64)=ccall((:gpuseed,libkunet),Void,(Culonglong,),n)


# For debugging
function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end

end	########## CUDA extensions
