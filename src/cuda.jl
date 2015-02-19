# CUDA extensions:
using CUDArt
using CUBLAS
typealias Cmat Ptr{Float32}
const libkunet = find_library(["libkunet"], ["."])

import Base: ctranspose         # TODO: these don't hang high enough in the type hierarchy
import InplaceOps: Transpose, mul!, badd!, bmul!, bsub! # TODO: non of these implementations are complete
ctranspose(x::Matrix)=Transpose(x)  # This was overwritten in base
ctranspose(x::CudaVecOrMat)=Transpose(x)
mul!(O::Matrix, A::Matrix, B::Transpose) = A_mul_Bt!(O,A,B.obj)   # 3rd arg B gives type error
mul!(O::Matrix, A::Transpose, B::Matrix) = At_mul_B!(O,A.obj,B)   # 2nd arg A gives type error
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::CudaVecOrMat) = CUBLAS.gemm!('N','N',one(eltype(O)),A,B,zero(eltype(O)),O)  # InplaceOps.jl:53
mul!(O::CudaVecOrMat, A::Transpose, B::CudaVecOrMat) = CUBLAS.gemm!('T','N',one(eltype(O)),A.obj,B,zero(eltype(O)),O)
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::Transpose) = CUBLAS.gemm!('N','T',one(eltype(O)),A,B.obj,zero(eltype(O)),O)
badd!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaVecOrMat) = ccall((:badd,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,B) # InplaceOps.jl:83
bmul!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, x::Float32) = CUBLAS.scal!(length(A), x, A, 1)
bsub!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaMatrix) = CUBLAS.axpy!(length(A), -1.0f0, B, 1, A, 1)

# # I could not get this to work:
# import Base: convert, promote_rule
# convert(::Type{Mat},x::Transpose{Mat})=x.obj
# promote_rule(::Type{Mat},::Type{Transpose{Mat}})=Mat

import Base: sum!, zeros  # TODO: add error checking here since this is not a full implementation of sum!
sum!(r::CudaVecOrMat, A::CudaMatrix) = ccall((:bsum,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,r) # reducedim.jl:226
zeros(A::CudaMatrix)=CUBLAS.scal!(length(A), zero(eltype(A)), copy(A), 1)

# For debugging
function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end
