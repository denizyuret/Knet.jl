import Base: *, transpose, permutedims, A_mul_B!
import Base: A_mul_Bt, A_mul_Bt!, A_mul_Bc, A_mul_Bc!
import Base: At_mul_B, At_mul_B!, Ac_mul_B, Ac_mul_B!
import Base: At_mul_Bt, At_mul_Bt!, Ac_mul_Bc, Ac_mul_Bc!
import Base.LinAlg.BLAS: gemm!
import Base.LinAlg: axpy!, scale!
export axpy!

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
At_mul_Bt{T}(A::KnetMatrix{T}, B::KnetMatrix{T})=At_mul_Bt!(similar(A,(size(A,2),size(B,1))),A,B)
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
    if T<:Float64
        @cuda(cublas, cublasDgemm_v2, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Ptr{T}, Cint, Ptr{T}, Ptr{T}, Cint), cublashandle(), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    elseif T<:Float32
        @cuda(cublas, cublasSgemm_v2, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Ptr{T}, Cint, Ptr{T}, Ptr{T}, Cint), cublashandle(), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    # elseif T<:Float16
    #     @cuda(cublas, cublasHgemm, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Ptr{T}, Cint, Ptr{T}, Ptr{T}, Cint), cublashandle(), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    else
        error("CUBLAS does not support $T")
    end
    return C
end

function axpy!{T}(n::Integer, alpha::Number, x::KnetArray{T}, incx::Integer, y::KnetArray{T}, incy::Integer)
    length(x) == length(y) || throw(DimensionMismatch("$(map(size,(x,y)))"))
    alpha = T[alpha]
    if T<:Float32
        @cuda(cublas, cublasSaxpy_v2, (Cptr, Cint, Ptr{T}, Ptr{T}, Cint, Ptr{T}, Cint), cublashandle(), n, alpha, x, incx, y, incy)
    elseif T<:Float64
        @cuda(cublas, cublasDaxpy_v2, (Cptr, Cint, Ptr{T}, Ptr{T}, Cint, Ptr{T}, Cint), cublashandle(), n, alpha, x, incx, y, incy)
    else
        error("$T not supported")
    end
    return y
end

axpy!{T}(alpha::Number, x::KnetArray{T}, y::KnetArray{T})=axpy!(length(x),alpha,x,1,y,1)


function scal!{T}(n::Integer, alpha::Number, x::KnetArray{T}, incx::Integer)
    alpha = T[alpha]
    if T<:Float32
        @cuda(cublas, cublasSscal_v2, (Cptr, Cint, Ptr{T}, Ptr{T}, Cint), cublashandle(), n, alpha, x, incx)
    elseif T<:Float64
        @cuda(cublas, cublasDscal_v2, (Cptr, Cint, Ptr{T}, Ptr{T}, Cint), cublashandle(), n, alpha, x, incx)
    else
        error("$T not supported")
    end
    return x
end

scale!{T}(alpha::Number, x::KnetArray{T})=scal!(length(x),alpha,x,1)
scale!{T}(x::KnetArray{T}, alpha::Number)=scal!(length(x),alpha,x,1)

function transpose{T}(x::KnetArray{T})
    ndims(x) != 2 && error("Transpose is supported only for 2D KnetArrays")
    sz = size(x)
    y = similar(x,(sz[2],sz[1]))
    if T<:Float32
        @cuda(cublas, cublasSgeam, (Cptr,UInt32,UInt32,Cint,Cint,Ptr{T},Ptr{T},Cint,Ptr{T},Ptr{T},Cint,Ptr{T},Cint),
              cublashandle(),1,1,size(y,1),size(y,2),Ref(T(1.0)),x,size(x,1),Ref(T(0.0)),x,size(x,1),y,size(y,1))
    elseif T<:Float64
        @cuda(cublas, cublasDgeam, (Cptr,UInt32,UInt32,Cint,Cint,Ptr{T},Ptr{T},Cint,Ptr{T},Ptr{T},Cint,Ptr{T},Cint),
              cublashandle(),1,1,size(y,1),size(y,2),Ref(T(1.0)),x,size(x,1),Ref(T(0.0)),x,size(x,1),y,size(y,1))
    else
        error("CUBLAS does not support $T")
    end
    return y
end


"""

    mat(x) 

Reshape x into a two-dimensional matrix.

This is typically used when turning the output of a 4-D convolution
result into a 2-D input for a fully connected layer.  For 1-D inputs
returns `reshape(x, (length(x),1))`.  For inputs with more than two
dimensions of size `(X1,X2,...,XD)`, returns

    reshape(x, (X1*X2*...*X[D-1],XD))

"""
function mat(x)
    if ndims(x) > 2
        xn = size(x,ndims(x))
        reshape(x, (div(length(x),xn),xn))
    elseif ndims(x)==2
        x
    elseif ndims(x)==1
        reshape(x, (length(x),1))
    else
        throw(MethodError(mat,x))
    end
end


import Base: permutedims, ipermutedims
function permutedims{T,N}(x::KnetArray{T,N}, dims)
    if length(dims) != N; throw(DimensionMismatch()); end
    if N == 2
        funcName = permutefunc(x,dims)
        y = similar(x, size(x,dims[1]), size(x,dims[2]))
        @eval ccall(($funcName,libknet8),Void,(Ptr{$T},Cint,Cint,Ptr{$T},Cint,Cint),
                    $x,size($x,1),size($x,2),$y,size($y,1),size($y,2))
        return y
    elseif N == 3
        funcName = permutefunc(x,dims)
        y = similar(x, size(x,dims[1]), size(x,dims[2]), size(x,dims[3]))
        @eval ccall(($funcName,libknet8),Void,(Ptr{$T},Cint,Cint,Cint,Ptr{$T},Cint,Cint,Cint),
                    $x,size($x,1),size($x,2),size($x,3),$y,size($y,1),size($y,2),size($y,3))
        return y
    elseif N == 4
        funcName = permutefunc(x,dims)
        y = similar(x, size(x,dims[1]), size(x,dims[2]), size(x,dims[3]), size(x,dims[4]))
        @eval ccall(($funcName,libknet8),Void,(Ptr{$T},Cint,Cint,Cint,Cint,Ptr{$T},Cint,Cint,Cint,Cint),
                    $x,size($x,1),size($x,2),size($x,3),size($x,4),$y,size($y,1),size($y,2),size($y,3),size($y,4))
        return y
    elseif N == 5
        funcName = permutefunc(x,dims)
        y = similar(x, size(x,dims[1]), size(x,dims[2]), size(x,dims[3]), size(x,dims[4]), size(x,dims[5]))
        @eval ccall(($funcName,libknet8),Void,(Ptr{$T},Cint,Cint,Cint,Cint,Cint,Ptr{$T},Cint,Cint,Cint,Cint,Cint),
                    $x,size($x,1),size($x,2),size($x,3),size($x,4),size($x,5),$y,size($y,1),size($y,2),size($y,3),size($y,4),size($y,5))
        return y
    else
        error("Unsupported number of dimensions")
    end
end

function permutefunc{T,N}(x::KnetArray{T,N}, dims)
    funcName = "permutedims_$(N)D_"
    for i=1:N
        funcName = funcName * "$(dims[i])_"
    end
    if T<:Float32
        funcName = funcName * "32"
    elseif T<:Float64
        funcName = funcName * "64"
    else
        error("$T not supported")
    end
    return funcName
end    

function ipermutedims(A::KnetArray,perm)
    iperm = Array{Int}(length(perm))
    for (i,p) = enumerate(perm)
        iperm[p] = i
    end
    return permutedims(A,iperm)
end

# Low level gemm! call with pointers

using Base.LinAlg
using Base.LinAlg.BLAS: libblas, BlasInt
using Compat: @blasfunc

# C := alpha*op(A)*op(B) + beta*C, where:
# op(X) is one of op(X) = X, or op(X) = XT, or op(X) = XH,
# alpha and beta are scalars,
# A, B and C are matrices:
# op(A) is an m-by-k matrix,
# op(B) is a k-by-n matrix,
# C is an m-by-n matrix.

for (gemm, elty) in ((:dgemm_,:Float64), (:sgemm_,:Float32))
    @eval begin
        function gemm!(transA::Char, transB::Char, M::Int, N::Int, K::Int, alpha::($elty), A::Ptr{$elty}, B::Ptr{$elty}, beta::($elty), C::Ptr{$elty})
            if transA=='N'; lda=M; else; lda=K; end
            if transB=='N'; ldb=K; else; ldb=N; end
            ldc = M;
            ccall((@blasfunc($gemm), libblas), Void,
                  (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty},
                   Ptr{BlasInt}),
                  &transA, &transB, &M, &N, &K,
                  &alpha, A, &lda, B, &ldb, &beta, C, &ldc)
        end
    end
end
