import Base: *, A_mul_B!
import Base: A_mul_Bt, A_mul_Bt!, A_mul_Bc, A_mul_Bc!
import Base: At_mul_B, At_mul_B!, Ac_mul_B, Ac_mul_B!
import Base: At_mul_Bt, At_mul_Bt!, Ac_mul_Bc, Ac_mul_Bc!
import Base.LinAlg.BLAS: gemm!
import Base.LinAlg: axpy!

A_mul_B!{T}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=gemm!('N','N',one(T),A,B,zero(T),C)
(*){T}(A::KnetMatrix{T},B::KnetMatrix{T})=A_mul_B!(similar(A,(size(A,1),size(B,2))),A,B)

A_mul_Bt!{T}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=gemm!('N','T',one(T),A,B,zero(T),C)
A_mul_Bt{T}(A::KnetMatrix{T}, B::KnetMatrix{T})=A_mul_Bt!(similar(A,(size(A,1),size(B,1))),A,B)
A_mul_Bc!{T<:Real}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=A_mul_Bt!(C,A,B)
A_mul_Bc{T<:Real}(A::KnetMatrix{T}, B::KnetMatrix{T})=A_mul_Bt(A,B)

At_mul_B!{T}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=gemm!('T','N',one(T),A,B,zero(T),C)
At_mul_B{T}(A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_B!(similar(A,(size(A,2),size(B,2))),A,B)
Ac_mul_B!{T<:Real}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_B!(C,A,B)
Ac_mul_B{T<:Real}(A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_B(A,B)

At_mul_Bt!{T}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=gemm!('T','T',one(T),A,B,zero(T),C)
At_mul_Bt{T}(A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_Bt!(similar(A,(size(A,2),size(B,2))),A,B)
Ac_mul_Bc!{T<:Real}(C::KnetMatrix{T}, A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_Bt!(C,A,B)
Ac_mul_Bc{T<:Real}(A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_Bt(A,B)


function gemm!{T}(transA::Char, transB::Char, alpha::Number, A::KnetArray{T}, B::KnetArray{T}, beta::Number, C::KnetArray{T})
    cublasop(c::Char)=(if c=='N'; 0; elseif c=='T'; 1; elseif c=='C'; 2; else error("Unknown cublas op $c"); end)
    size2(x,i)=(if ndims(x)<=2; size(x,i); elseif i==1; div(length(x),size(x,ndims(x))); elseif i==2; size(x,ndims(x)); else 1; end)
    if transA == 'N'
        m=size2(A,1); k=size2(A,2)
    else
        m=size2(A,2); k=size2(A,1)
    end
    if transB == 'N'
        n=size2(B,2); k==size2(B,1) || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    else
        n=size2(B,1); k==size2(B,2) || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    end
    (m == size2(C,1) && n == size(C,2)) || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    transa = cublasop(transA); transb = cublasop(transB)
    alpha = T[alpha]; beta = T[beta]
    lda = size2(A,1); ldb = size2(B,1); ldc = size2(C,1)
    if T<:Float32
        @cuda(cublas, cublasSgemm_v2, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Ptr{T}, Cint, Ptr{T}, Ptr{T}, Cint), cublashandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    elseif T<:Float64
        @cuda(cublas, cublasDgemm_v2, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Ptr{T}, Cint, Ptr{T}, Ptr{T}, Cint), cublashandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    else
        error("CUBLAS does not support $T")
    end
    return C
end

function axpy!{T}(n::Integer, alpha::Number, x::KnetArray{T}, incx::Integer, y::KnetArray{T}, incy::Integer)
    length(x) == length(y) || throw(DimensionMismatch("$(map(size,(x,y)))"))
    alpha = T[alpha]
    if T<:Float32
        @cuda(cublas, cublasSaxpy_v2, (Cptr, Cint, Ptr{T}, Ptr{T}, Cint, Ptr{T}, Cint), cublashandle, n, alpha, x, incx, y, incy)
    elseif T<:Float64
        @cuda(cublas, cublasDaxpy_v2, (Cptr, Cint, Ptr{T}, Ptr{T}, Cint, Ptr{T}, Cint), cublashandle, n, alpha, x, incx, y, incy)
    else
        error("$T not supported")
    end
    return y
end

axpy!{T}(alpha::Number, x::KnetArray{T}, y::KnetArray{T})=axpy!(length(x),alpha,x,1,y,1)

export axpy!
