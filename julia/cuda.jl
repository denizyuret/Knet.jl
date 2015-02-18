# CUDA extensions:
using CUDArt
using CUBLAS
typealias Cmat Ptr{Float32}
const libjnet = find_library(["libjnet"], ["."])
import InplaceOps: mul!, badd!
mul!(O::CudaVecOrMat, A::CudaVecOrMat, B::CudaVecOrMat) = CUBLAS.gemm!('N','N',one(eltype(A)),A,B,zero(eltype(A)),O)  # InplaceOps.jl:53
badd!(::Type{InplaceOps.Inplace{1}}, A::CudaMatrix, B::CudaVecOrMat) = ccall((:badd,libjnet),Void,(Cint,Cint,Cmat,Cmat),size(A,1),size(A,2),A,B) # InplaceOps.jl:83
reluforw(y::CudaArray)=ccall((:reluforw,libjnet),Void,(Cint,Cmat),length(y),y)
reluback(y::CudaArray,dy::CudaArray)=ccall((:reluback,libjnet),Void,(Cint,Cmat,Cmat),length(y),y,dy)

import Base: ctranspose
import InplaceOps: Transpose, AbstractVMF
typealias Mat Array{Float32,2}
ctranspose(x::Mat)=Transpose(x)  # This was overwritten in base
mul!(O::AbstractVMF, A::AbstractVMF, B::Transpose) = A_mul_Bt!(O,A,B.obj)   # 3rd arg B gives type error
mul!(O::AbstractVMF, A::Transpose, B::AbstractVMF) = At_mul_B!(O,A.obj,B)   # 2nd arg A gives type error

# # I could not get this to work:
# import Base: convert, promote_rule
# convert(::Type{Mat},x::Transpose{Mat})=x.obj
# promote_rule(::Type{Mat},::Type{Transpose{Mat}})=Mat
