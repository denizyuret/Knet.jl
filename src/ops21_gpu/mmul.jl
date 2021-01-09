import Knet.Ops21: mmul
using Knet.KnetArrays: DevArray
using CUDA.CUBLAS: CUBLAS, cublasDgemm_v2, cublasSgemm_v2, cublasOperation_t
using AutoGrad: AutoGrad, @primitive1

function mmul(x1::R, x2::R; dims=1) where {T,R<:DevArray{T}}
    @assert ndims(x1) > dims "ndims(w) must be > dims"
    ntuple(i->size(x1,ndims(x1)-dims+i),dims) === ntuple(i->size(x2,i),dims) || throw(DimensionMismatch("mmul: w=$(size(x1)) x=$(size(x2)) dims=$dims"))
    t1,t2 = cublasOperation_t.((0,0))
    msize = (size(x1,i) for i in 1:ndims(x1)-dims)
    nsize = (size(x2,i) for i in 1:dims)
    ksize = (size(x2,i) for i in dims+1:ndims(x2))
    m,n,k = prod.((msize, nsize, ksize))
    y = similar(x1, (msize..., ksize...))
    if T<:Float64
        cublasDgemm_v2(CUBLAS.handle(), t1, t2, m, n, k, 1, x1, m, x2, n, 0, y, m)
    elseif T<:Float32
        cublasSgemm_v2(CUBLAS.handle(), t1, t2, m, n, k, 1, x1, m, x2, n, 0, y, m)
    else
        error("CUBLAS does not support $T")
    end
    return y
end

function _mmul1(x1::R, x2::R; dims=1) where {T,R<:DevArray{T}} # x1 transposed
    @assert ntuple(i->size(x1,i),dims) === ntuple(i->size(x2,i),dims)
    t1,t2 = cublasOperation_t.((1,0))
    msize = (size(x1,i) for i in dims+1:ndims(x1))
    nsize = (size(x1,i) for i in 1:dims)
    ksize = (size(x2,i) for i in dims+1:ndims(x2))
    m,n,k = prod.((msize, nsize, ksize))
    y = similar(x1, (msize..., ksize...))
    if T<:Float64
        cublasDgemm_v2(CUBLAS.handle(), t1, t2, m, n, k, 1, x1, m, x2, n, 0, y, m)
    elseif T<:Float32
        cublasSgemm_v2(CUBLAS.handle(), t1, t2, m, n, k, 1, x1, m, x2, n, 0, y, m)
    else
        error("CUBLAS does not support $T")
    end
    return y
end

function _mmul2(x1::R, x2::R; dims=1) where {T,R<:DevArray{T}} # x2 transposed
    @assert ndims(x1) > dims && ndims(x2) > dims
    @assert ntuple(i->size(x1,ndims(x1)-dims+i),dims) === ntuple(i->size(x2,ndims(x2)-dims+i),dims)
    t1,t2 = cublasOperation_t.((0,1))
    msize = (size(x1,i) for i in 1:ndims(x1)-dims)
    nsize = (size(x1,i) for i in 1+ndims(x1)-dims:ndims(x1))
    ksize = (size(x2,i) for i in 1:ndims(x2)-dims)
    m,n,k = prod.((msize, nsize, ksize))
    y = similar(x1, (msize..., ksize...))
    if T<:Float64
        cublasDgemm_v2(CUBLAS.handle(), t1, t2, m, n, k, 1, x1, m, x2, n, 0, y, m)
    elseif T<:Float32
        cublasSgemm_v2(CUBLAS.handle(), t1, t2, m, n, k, 1, x1, m, x2, n, 0, y, m)
    else
        error("CUBLAS does not support $T")
    end
    return y
end

function _mmul3(x1::R, x2::R; dims=1) where {T,R<:DevArray{T}} # x1,x2 transposed
    @assert ndims(x2) > dims
    @assert ntuple(i->size(x1,i),dims) === ntuple(i->size(x2,ndims(x2)-dims+i),dims)
    t1,t2 = cublasOperation_t.((1,1))
    msize = (size(x1,i) for i in dims+1:ndims(x1))
    nsize = (size(x1,i) for i in 1:dims)
    ksize = (size(x2,i) for i in 1:ndims(x2)-dims)
    m,n,k = prod.((msize, nsize, ksize))
    y = similar(x1, (msize..., ksize...))
    if T<:Float64
        cublasDgemm_v2(CUBLAS.handle(), t1, t2, m, n, k, 1, x1, m, x2, n, 0, y, m)
    elseif T<:Float32
        cublasSgemm_v2(CUBLAS.handle(), t1, t2, m, n, k, 1, x1, m, x2, n, 0, y, m)
    else
        error("CUBLAS does not support $T")
    end
    return y
end

@primitive1   mmul(x1,x2;dims=1),dy  _mmul2(dy,x2; dims=ndims(x2)-dims)  _mmul1(x1,dy; dims=ndims(x1)-dims)
@primitive1 _mmul1(x1,x2;dims=1),dy  _mmul2(x2,dy; dims=ndims(x2)-dims)    mmul(x1,dy; dims=ndims(x1)-dims)
@primitive1 _mmul2(x1,x2;dims=1),dy    mmul(dy,x2; dims=ndims(x2)-dims)  _mmul1(dy,x1; dims=ndims(x1)-dims)
@primitive1 _mmul3(x1,x2;dims=1),dy  _mmul3(x2,dy; dims=ndims(x2)-dims)  _mmul3(dy,x1; dims=ndims(x1)-dims)
