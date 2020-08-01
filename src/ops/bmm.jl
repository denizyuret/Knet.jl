using Base: has_offset_axes, unsafe_convert
using LinearAlgebra: chkstride1, BlasInt
using LinearAlgebra.BLAS: libblas, @blasfunc


"""
    bmm(A, B ; transA=false, transB=false)
Perform a batch matrix-matrix product of matrices stored in `A` and `B`. size(A,2) ==
size(B,1) and size(A)[3:end] and size(B)[3:end] must match.
If A is a (m,n,b...) tensor, B is a (n,k,b...) tensor, and the output is a (m,k,b...)
tensor.
"""
function bmm(A::AbstractArray{T}, B::AbstractArray{T}; transA::Bool = false, transB::Bool = false) where T
    sa, sb = size(A), size(B)
    m, k   = transA ? (sa[2],sa[1]) : (sa[1],sa[2])
    kb, n  = transB ? (sb[2],sb[1]) : (sb[1],sb[2])
    @assert kb == k && sa[3:end]==sb[3:end]
    a3, b3 = reshape(A,sa[1],sa[2],:), reshape(B,sb[1],sb[2],:)
    C = similar(A, m, n, size(a3,3))
    bmm!((transA ? 'T' : 'N'), (transB ? 'T' : 'N'), one(T), a3, b3, zero(T), C)
    reshape(C,m,n,sb[3:end]...)
end


for (gemm, elty) in
    ((:dgemm_,:Float64),
     (:sgemm_,:Float32),)
    @eval begin
        function bmm!(transA::AbstractChar,
                      transB::AbstractChar,
                      alpha::($elty),
                      A::AbstractArray{$elty, 3},
                      B::AbstractArray{$elty, 3},
                      beta::($elty),
                      C::AbstractArray{$elty, 3})
            @assert !has_offset_axes(A, B, C)
            @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
            end
            chkstride1(A)
            chkstride1(B)
            chkstride1(C)

            ptrA = unsafe_convert(Ptr{$elty}, A)
            ptrB = unsafe_convert(Ptr{$elty}, B)
            ptrC = unsafe_convert(Ptr{$elty}, C)

            for k in 1:size(A, 3)
                ccall((@blasfunc($gemm), libblas), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                     Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
                     Ref{BlasInt}),
                     transA, transB, m, n,
                     ka, alpha, ptrA, max(1,stride(A,2)),
                     ptrB, max(1,stride(B,2)), beta, ptrC,
                     max(1,stride(C,2)))

                ptrA += size(A, 1) * size(A, 2) * sizeof($elty)
                ptrB += size(B, 1) * size(B, 2) * sizeof($elty)
                ptrC += size(C, 1) * size(C, 2) * sizeof($elty)
            end

            C
        end
    end
end

@primitive bmm(x1,x2; transA::Bool=false, transB::Bool=false),dy,y (transA ? bmm(x2, dy; transA=transB , transB=true) :  bmm(dy, x2;  transA=false, transB=!transB) )    (transB ? bmm(dy,x1; transA=true , transB=transA) :  bmm(x1, dy;  transA=!transA , transB=false))
@zerograd  bmm!(transA::AbstractChar, transB::AbstractChar, alpha::Number, A, B, beta::Number, C)
