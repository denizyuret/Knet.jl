using CUDArt
using CUBLAS: gemm!

A_mul_B!{T}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=gemm!('N','N',one(T),A,B,zero(T),C)
(*){T}(A::KnetMatrix{T},B::KnetMatrix{T})=A_mul_B!(tmplike(A,(size(A,1),size(B,2))),A,B)

A_mul_Bt!{T}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=gemm!('N','T',one(T),A,B,zero(T),C)
A_mul_Bt{T}(A::KnetMatrix{T}, B::KnetMatrix{T})=A_mul_Bt!(tmplike(A,(size(A,1),size(B,1))),A,B)
A_mul_Bc!{T<:Real}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=A_mul_Bt!(C,A,B)
A_mul_Bc{T<:Real}(A::KnetMatrix{T}, B::KnetMatrix{T})=A_mul_Bt(A,B)

At_mul_B!{T}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=gemm!('T','N',one(T),A,B,zero(T),C)
At_mul_B{T}(A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_B!(tmplike(A,(size(A,2),size(B,2))),A,B)
Ac_mul_B!{T<:Real}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_B!(C,A,B)
Ac_mul_B{T<:Real}(A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_B(A,B)

At_mul_Bt!{T}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=gemm!('T','T',one(T),A,B,zero(T),C)
At_mul_Bt{T}(A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_Bt!(tmplike(A,(size(A,2),size(B,2))),A,B)
Ac_mul_Bc!{T<:Real}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_Bt!(C,A,B)
Ac_mul_Bc{T<:Real}(A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_Bt(A,B)

