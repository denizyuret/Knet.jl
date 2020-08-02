import ..Ops20: bmm, bmm!

function bmm(A::CuArray{T}, B::CuArray{T}; transA::Bool = false, transB::Bool = false) where {T}
    sa, sb = size(A), size(B)
    m, k   = transA ? (sa[2],sa[1]) : (sa[1],sa[2])
    kb, n  = transB ? (sb[2],sb[1]) : (sb[1],sb[2])
    @assert kb == k && sa[3:end]==sb[3:end]
    a3, b3 = reshape(A,sa[1],sa[2],:), reshape(B,sb[1],sb[2],:)
    C = similar(A, m, n, size(a3,3))    
    bmm!((transA ? 'T' : 'N'), (transB ? 'T' : 'N'), one(T), a3, b3, zero(T), C)
    reshape(C,m,n,sb[3:end]...)
end 

function bmm!(transA::AbstractChar, transB::AbstractChar, alpha::Number, A::CuArray{T}, B::CuArray{T}, beta::Number, C::CuArray{T}) where {T}
    cublasop(c::Char)=CUBLAS.cublasOperation_t(if c==='N'; 0; elseif c==='T'; 1; elseif c==='C'; 2; else error("Unknown cublas op $c"); end)
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
    alpha = T[alpha]; beta = T[beta]
    strideA, strideB, strideC = m*k, k*n, m*n
    if T<:Float64
        # @cublas(cublasDgemmStridedBatched, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Clonglong, Ptr{T}, Cint, Clonglong, Ptr{T}, Ptr{T}, Cint, Clonglong, Cint), cublashandle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
        CUBLAS.cublasDgemmStridedBatched(CUBLAS.handle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
    elseif T<:Float32
        # @cublas(cublasSgemmStridedBatched, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Clonglong, Ptr{T}, Cint, Clonglong, Ptr{T}, Ptr{T}, Cint, Clonglong, Cint), cublashandle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
        CUBLAS.cublasSgemmStridedBatched(CUBLAS.handle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
    else
        error("CUBLAS does not support $T")
    end
    return C
end
