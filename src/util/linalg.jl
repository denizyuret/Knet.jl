import Base: A_mul_B!, A_mul_Bt!, At_mul_B!
using Base.LinAlg.BLAS: gemm! # , axpy!, scal!

### CUDAARRAY

A_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','N',one(T),A,B,zero(T),C)
A_mul_Bt!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','T',one(T),A,B,zero(T),C)
At_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('T','N',one(T),A,B,zero(T),C)

### KUDENSE

# The input could be a tensor or a vector.  In which case perform
# internal calculations in 2D.

mat2d(x)=(ndims(x)==2 ? x : reshape(x, size2(x)))
A_mul_B!{S,T}(C::KUdense{S,T}, A::KUparam{S,T}, B::KUdense{S,T})=(A_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
At_mul_B!{S,T}(C::KUdense{S,T}, A::KUparam{S,T}, B::KUdense{S,T})=(At_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
A_mul_Bt!{S,T}(C::Union(Array{T},CudaArray{T}), A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_Bt!(mat2d(C), mat2d(A.arr), mat2d(B.arr)); C)

### KUSPARSE
# TODO: fix sparse stuff

# At_mul_B!(k::Matrix, x::SparseMatrixCSC, s::SparseMatrixCSC)=A_mul_B!(k,x',s)

# function A_mul_B!(k::Matrix, x::SparseMatrixCSC, s::SparseMatrixCSC) # 1607
#     @assert size(k)==(size(x,1), size(s,2))
#     fill!(k, zero(eltype(k)))
#     @inbounds @simd for scol=1:size(s,2)
#         @inbounds @simd for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
#             srow = s.rowval[sp]
#             sval = s.nzval[sp]  # 133
#             @inbounds @simd for xp=x.colptr[srow]:(x.colptr[srow+1]-1)
#                 xrow = x.rowval[xp] # 63
#                 xval = x.nzval[xp]  # 217
#                 yinc = xval * sval  # 245
#                 k[xrow,scol] += yinc # 789
#             end
#         end
#     end
#     return k
# end

# function A_mul_B!(k::Matrix, x::Matrix, s::SparseMatrixCSC) # 1607
#     @assert size(k)==(size(x,1), size(s,2))
#     fill!(k, zero(eltype(k)))
#     @inbounds @simd for scol=1:size(s,2)
#         @inbounds @simd for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
#             sval = s.nzval[sp]  # 133
#             srow = s.rowval[sp] # xcol
#             @inbounds @simd for xrow=1:size(x,1)
#                 xval = x[xrow,srow]
#                 yinc = xval * sval  # 245
#                 k[xrow,scol] += yinc # 789
#             end
#         end
#     end
#     return k
# end

# ### TODO: fix the sparse stuff

# # At_mul_B!{T}(k::AbstractCudaMatrix{T}, x::CudaSparseMatrixCSC{T}, s::CudaSparseMatrixCSC{T})=A_mul_B!(k,x.',s)

# function At_mul_B!(k::AbstractCudaMatrix{Float32}, x::CudaSparseMatrixCSC{Float32}, s::CudaSparseMatrixCSC{Float32})
#     @assert size(k)==(size(x,2),size(s,2))
#     ccall((:At_mul_B_32,libkunet),Void,
#           (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
#           size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
#     gpusync()
#     return k
# end

# function At_mul_B!(k::AbstractCudaMatrix{Float64}, x::CudaSparseMatrixCSC{Float64}, s::CudaSparseMatrixCSC{Float64})
#     @assert size(k)==(size(x,2),size(s,2))
#     ccall((:At_mul_B_64,libkunet),Void,
#           (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
#           size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
#     gpusync()
#     return k
# end

# function A_mul_B!(k::AbstractCudaMatrix{Float32}, x::CudaSparseMatrixCSC{Float32}, s::CudaSparseMatrixCSC{Float32})
#     @assert size(k)==(size(x,1),size(s,2))
#     ccall((:A_mul_B_32,libkunet),Void,
#           (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
#           size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
#     gpusync()
#     return k
# end

# function A_mul_B!(k::AbstractCudaMatrix{Float64}, x::CudaSparseMatrixCSC{Float64}, s::CudaSparseMatrixCSC{Float64})
#     @assert size(k)==(size(x,1),size(s,2))
#     ccall((:A_mul_B_64,libkunet),Void,
#           (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
#           size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
#     gpusync()
#     return k
# end

