import Base: A_mul_B!, A_mul_Bt!, At_mul_B!
import Base.LinAlg.BLAS: gemm! # , axpy!, scal!

# A_mul_B!{S,T}(C::KUdense{S,T,2}, A::KUparam{S,T,2}, B::KUdense{S,T,2})=(A_mul_B!(C.arr, A.arr, B.arr); C)
# A_mul_Bt!{S,T}(C::Union(Array{T,2},CudaArray{T,2}), A::KUdense{S,T,2}, B::KUdense{S,T,2})=(A_mul_Bt!(C, A.arr, B.arr); C)
# At_mul_B!{S,T}(C::KUdense{S,T,2}, A::KUparam{S,T,2}, B::KUdense{S,T,2})=(At_mul_B!(C.arr, A.arr, B.arr); C)

A_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','N',one(T),A,B,zero(T),C)
A_mul_Bt!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','T',one(T),A,B,zero(T),C)
At_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('T','N',one(T),A,B,zero(T),C)

# The input could be a tensor or a vector.  In which case perform
# internal calculations in 2D: actually extend linalg routines to
# handle this:

mat2d(x)=(ndims(x)==2 ? x : reshape(x, size2(x)))

A_mul_B!{S,T}(C::KUdense{S,T}, A::KUparam{S,T}, B::KUdense{S,T})=(A_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
At_mul_B!{S,T}(C::KUdense{S,T}, A::KUparam{S,T}, B::KUdense{S,T})=(At_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
A_mul_Bt!{S,T}(C::Union(Array{T},CudaArray{T}), A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_Bt!(mat2d(C), mat2d(A.arr), mat2d(B.arr)); C)

# At_mul_B!(C,A,B)=(At_mul_B!(mat2d(C),mat2d(A),mat2d(B));C)
# A_mul_B!(C,A,B)=(A_mul_B!(mat2d(C),mat2d(A),mat2d(B));C)
# A_mul_Bt!(C,A,B)=(A_mul_Bt!(mat2d(C),mat2d(A),mat2d(B));C)

# function At_mul_B!(C,A,B)
#     @show 1
#     @show map(summary, (C,A,B))
#     @show map(pointer, (C,A,B))
#     (AA,BB,CC)=map(mat2d, (A,B,C))
#     @show map(summary, (CC,AA,BB))
#     @show map(pointer, (CC,AA,BB))
#     At_mul_B!(CC,AA,BB)
#     info("we did it.")
#     return C
# end

