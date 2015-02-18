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
