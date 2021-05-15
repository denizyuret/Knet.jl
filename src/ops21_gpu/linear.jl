import Knet.Ops21: linear
using Knet.KnetArrays: DevArray
using CUDA.CUBLAS: CUBLAS, cublasDgemm_v2, cublasSgemm_v2, CUBLAS_OP_T, CUBLAS_OP_N
using AutoGrad: AutoGrad, @primitive1

# op(A) m × k , op(B) k × n and C m × n; lda, ldb, ldc leading dimensions

for (T,F) in ((Float32,cublasSgemm_v2), (Float64,cublasDgemm_v2)); @eval begin

    function linear(A::DevArray{$T}, B::DevArray{$T}; dims=1)
        @assert ndims(A) > dims "ndims(A) must be > dims"
        ntuple(i->size(A,ndims(A)-dims+i),dims) === ntuple(i->size(B,i),dims) || throw(DimensionMismatch("linear: w=$(size(A)) x=$(size(B)) dims=$dims"))
        transa,transb = (CUBLAS_OP_N,CUBLAS_OP_N)
        msize = (size(A,i) for i in 1:ndims(A)-dims)
        ksize = (size(B,i) for i in 1:dims)
        nsize = (size(B,i) for i in dims+1:ndims(B))
        m,n,k = prod.((msize, nsize, ksize))
        lda,ldb,ldc = m,k,m
        C = similar(A, (msize..., nsize...))
        $F(CUBLAS.handle(), transa, transb, m, n, k, 1, A, lda, B, ldb, 0, C, ldc)
        return C
    end

    function _linear1(A::DevArray{$T}, B::DevArray{$T}; dims)  # A transposed
        @assert ntuple(i->size(A,i),dims) === ntuple(i->size(B,i),dims)
        transa,transb = (CUBLAS_OP_T,CUBLAS_OP_N)
        msize = (size(A,i) for i in dims+1:ndims(A))
        ksize = (size(B,i) for i in 1:dims)
        nsize = (size(B,i) for i in dims+1:ndims(B))
        m,n,k = prod.((msize, nsize, ksize))
        lda,ldb,ldc = k,k,m
        C = similar(A, (msize..., nsize...))
        $F(CUBLAS.handle(), transa, transb, m, n, k, 1, A, lda, B, ldb, 0, C, ldc)
        return C
    end

    function _linear2(A::DevArray{$T}, B::DevArray{$T}; dims)  # B transposed
        @assert ndims(A) > dims && ndims(B) > dims
        @assert ntuple(i->size(A,ndims(A)-dims+i),dims) === ntuple(i->size(B,ndims(B)-dims+i),dims)
        transa,transb = (CUBLAS_OP_N,CUBLAS_OP_T)
        msize = (size(A,i) for i in 1:ndims(A)-dims)
        ksize = (size(A,i) for i in 1+ndims(A)-dims:ndims(A))
        nsize = (size(B,i) for i in 1:ndims(B)-dims)
        m,n,k = prod.((msize, nsize, ksize))
        lda,ldb,ldc = m,n,m
        C = similar(A, (msize..., nsize...))
        $F(CUBLAS.handle(), transa, transb, m, n, k, 1, A, lda, B, ldb, 0, C, ldc)
        return C
    end

    function _linear3(A::DevArray{$T}, B::DevArray{$T}; dims)  # A,B transposed
        @assert ndims(B) > dims
        @assert ntuple(i->size(A,i),dims) === ntuple(i->size(B,ndims(B)-dims+i),dims)
        transa,transb = (CUBLAS_OP_T,CUBLAS_OP_T)
        msize = (size(A,i) for i in dims+1:ndims(A))
        ksize = (size(A,i) for i in 1:dims)
        nsize = (size(B,i) for i in 1:ndims(B)-dims)
        m,n,k = prod.((msize, nsize, ksize))
        lda,ldb,ldc = k,n,m
        C = similar(A, (msize..., nsize...))
        $F(CUBLAS.handle(), transa, transb, m, n, k, 1, A, lda, B, ldb, 0, C, ldc)
        return C
    end
end; end


@primitive1   linear(A::DevArray,B::DevArray;dims=1),dy  _linear2(dy,B; dims=ndims(B)-dims)  _linear1(A,dy; dims=ndims(A)-dims)
@primitive1 _linear1(A::DevArray,B::DevArray;dims=1),dy  _linear2(B,dy; dims=ndims(B)-dims)    linear(A,dy; dims=ndims(A)-dims)
@primitive1 _linear2(A::DevArray,B::DevArray;dims=1),dy    linear(dy,B; dims=ndims(B)-dims)  _linear1(dy,A; dims=ndims(A)-dims)
@primitive1 _linear3(A::DevArray,B::DevArray;dims=1),dy  _linear3(B,dy; dims=ndims(B)-dims)  _linear3(dy,A; dims=ndims(A)-dims)
