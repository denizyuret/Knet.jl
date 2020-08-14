import Knet.Ops20: bmm, bmm!
using Knet.KnetArrays: DevArray
using CUDA.CUBLAS: CUBLAS, cublasSgemmStridedBatched, cublasDgemmStridedBatched, cublasOperation_t #, handle

function bmm(A::R, B::R; transA::Bool = false, transB::Bool = false) where {R<:DevArray}
    sa, sb = size(A), size(B)
    m, k   = transA ? (sa[2],sa[1]) : (sa[1],sa[2])
    kb, n  = transB ? (sb[2],sb[1]) : (sb[1],sb[2])
    @assert kb == k && sa[3:end]==sb[3:end]
    a3, b3 = reshape(A,sa[1],sa[2],:), reshape(B,sb[1],sb[2],:)
    C = similar(A, m, n, size(a3,3))    
    bmm!((transA ? 'T' : 'N'), (transB ? 'T' : 'N'), 1, a3, b3, 0, C)
    reshape(C,m,n,sb[3:end]...)
end 

function bmm!(transA::AbstractChar, transB::AbstractChar, alpha::Number, A::R, B::R, beta::Number, C::R) where {R<:DevArray}
    cublasop(c::Char)=cublasOperation_t(if c==='N'; 0; elseif c==='T'; 1; elseif c==='C'; 2; else error("Unknown cublas op $c"); end)
    if ndims(A) != 3 || ndims(B) != 3
        throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    end
    ma,ka,bsa = size(A)
    kb,nb,bsb = size(B)
    if bsa != bsb
        throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    end
    bs = bsa
    if transA == 'N'
        m=ma; k=ka;
    else
        m=ka; k=ma;
    end
    if transB == 'N'
        k == kb || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
        n=nb; k==kb;
    else
        k == nb || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
        n=kb; k==nb;
    end
    (m == size(C,1) && n == size(C,2) && bs == size(C,3)) || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    lda,ldb,ldc= ma,kb,m
    transa = cublasop(transA); transb = cublasop(transB)
    T = eltype(A)
    alpha = T[alpha]; beta = T[beta]
    strideA, strideB, strideC = m*k, k*n, m*n
    if T<:Float64
        cublasDgemmStridedBatched(CUBLAS.handle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
    elseif T<:Float32
        cublasSgemmStridedBatched(CUBLAS.handle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
    else
        error("cublasXgemmStridedBatched does not support $T")
    end
    return C
end
