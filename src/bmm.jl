"""

    bmm(A, B)

Perform a batch matrix-matrix product of matrices stored in `A` and `B`. size(A,2) ==
size(B,1) and size(A)[3:end] and size(B)[3:end] must match.

If A is a (m,n,b...) tensor, B is a (n,k,b...) tensor, and the output is a (m,k,b...)
tensor.

"""
function bmm(a, b)
    sa,sb = size(a),size(b)
    @assert sa[2] == sb[1] && sa[3:end] == sb[3:end]
    a3,b3 = reshape(a,sa[1],sa[2],:), reshape(b,sb[1],sb[2],:)
    c3 = similar(a,sa[1],sb[2],size(a3,3))
    bmm!(a3, b3, c3)
    reshape(c3, sa[1], sb[2:end]...)
end

function bmm(a::KnetArray{T}, b::KnetArray{T}) where T
    sa,sb = size(a),size(b)
    @assert sa[2] == sb[1] && sa[3:end] == sb[3:end]
    a3,b3 = reshape(a,sa[1],sa[2],:), reshape(b,sb[1],sb[2],:)
    c3 = similar(a,sa[1],sb[2],size(a3,3))
    bmm!('N','N',one(T),a3,b3,zero(T),c3)
    reshape(c3, sa[1], sb[2:end]...)
end

function bmm!(transA::AbstractChar, transB::AbstractChar, alpha::Number, A::KnetArray{T}, B::KnetArray{T}, beta::Number, C::KnetArray{T}) where {T}
    cublasop(c::Char)=(if c=='N'; 0; elseif c=='T'; 1; elseif c=='C'; 2; else error("Unknown cublas op $c"); end)
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
    lda,ldb,ldc=ma,ka,ma
    transa = cublasop(transA); transb = cublasop(transB)
    alpha = T[alpha]; beta = T[beta]
    strideA, strideB, strideC = m*k, k*n, m*n
    if T<:Float64
        @cublas(cublasDgemmStridedBatched, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Clonglong, Ptr{T}, Cint, Clonglong, Ptr{T}, Ptr{T}, Cint, Clonglong, Cint), cublashandle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
    elseif T<:Float32
        @cublas(cublasSgemmStridedBatched, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Clonglong, Ptr{T}, Cint, Clonglong, Ptr{T}, Ptr{T}, Cint, Clonglong, Cint), cublashandle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
    else
        error("CUBLAS does not support $T")
    end
    return C
end

function bmm!(A, B, C)
    if ndims(A) != 3 || ndims(B) != 3
        throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    end
    ma,ka,bsa = size(A)
    kb,nb,bsb = size(B)

    (bsa == bsb && ka == kb && size(C,1) == ma && size(C,2) == nb && size(C,3) == bsa) || throw(DimensionMismatch("$(map(size,(A,B,C)))"))

    for i=1:bsa
        C[:, :, i] = view(A, :, :, i) * view(B, :, :, i)
    end
    return C
end

@primitive bmm(x1,x2),dy,y  bmm(dy, permutedims(x2, (2,1,(3:ndims(x2))...)))  bmm(permutedims(x1, (2,1,(3:ndims(x1))...)), dy)
@zerograd bmm!(transA::AbstractChar, transB::AbstractChar, alpha::Number, A::KnetArray, B::KnetArray, beta::Number, C::KnetArray)
@zerograd bmm!(A, B, C)
