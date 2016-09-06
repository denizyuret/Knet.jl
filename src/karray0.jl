# Test the code with CudaArrays instead of KnetArrays:

using CUDArt,CUBLAS
typealias KnetArray{T,N} CudaArray{T,N}
typealias KnetMatrix{T} CudaMatrix{T}
typealias KnetVector{T} CudaVector{T}
typealias KnetPtr{T} CudaPtr{T}
