# CUDA extensions:
using CUDArt
using CUBLAS
typealias Cmat Ptr{Float32}
const libjnet = find_library(["libjnet"], ["."])
import InplaceOps: mul!, badd!
badd!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaVecOrMat) = ccall((:badd,libjnet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,B) # InplaceOps.jl:83
reluforw(y::CudaArray)=ccall((:reluforw,libjnet),Void,(Cint,Cmat),length(y),y)
reluback(y::CudaArray,dy::CudaArray)=ccall((:reluback,libjnet),Void,(Cint,Cmat,Cmat),length(y),y,dy)
softback(y::CudaArray,dy::CudaArray)=ccall((:softback,libjnet),Void,(Cint,Cint,Cmat,Cmat),size(y,1),size(y,2),y,dy)

import Base: ctranspose         # TODO: these don't hang high enough in the type hierarchy
import InplaceOps: Transpose, mul!
ctranspose(x::Matrix)=Transpose(x)  # This was overwritten in base
ctranspose(x::CudaVecOrMat)=Transpose(x)
mul!(O::Matrix, A::Matrix, B::Transpose) = A_mul_Bt!(O,A,B.obj)   # 3rd arg B gives type error
mul!(O::Matrix, A::Transpose, B::Matrix) = At_mul_B!(O,A.obj,B)   # 2nd arg A gives type error
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::CudaVecOrMat) = CUBLAS.gemm!('N','N',one(eltype(O)),A,B,zero(eltype(O)),O)  # InplaceOps.jl:53
mul!(O::CudaVecOrMat, A::Transpose, B::CudaVecOrMat) = CUBLAS.gemm!('T','N',one(eltype(O)),A.obj,B,zero(eltype(O)),O)
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::Transpose) = CUBLAS.gemm!('N','T',one(eltype(O)),A,B.obj,zero(eltype(O)),O)

# # I could not get this to work:
# import Base: convert, promote_rule
# convert(::Type{Mat},x::Transpose{Mat})=x.obj
# promote_rule(::Type{Mat},::Type{Transpose{Mat}})=Mat

import Base: sum!  # TODO: add error checking here since this is not a full implementation of sum!
sum!(r::CudaVecOrMat, A::CudaMatrix) = ccall((:bsum,libjnet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,r) # reducedim.jl:226
