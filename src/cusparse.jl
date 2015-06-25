import Base: size, similar, transpose, nnz, full, sparse

type CudaSparseMatrixCSC{Tv} <: AbstractCudaMatrix{Tv}
    m::Int                   # Number of rows
    n::Int                   # Number of columns
    colptr::CudaVector{Cint} # Column i is in colptr[i]+1:colptr[i+1], note that this is 0 based on cusparse
    rowval::CudaVector{Cint} # Row values of nonzeros
    nzval::CudaVector{Tv}    # Nonzero values
end

size(S::CudaSparseMatrixCSC) = (S.m, S.n)
size(S::CudaSparseMatrixCSC, d::Integer) = (d==1 ? S.m : d==2 ? S.n : error("Invalid index"))
nnz(S::CudaSparseMatrixCSC) = (to_host(S.colptr)[S.n+1]-1)

# cusparse can only handle Int32 indices
gpucopy(s::SparseMatrixCSC)=(t=CudaSparseMatrixCSC(s.m,s.n,CudaArray(int32(s.colptr)),CudaArray(int32(s.rowval)),CudaArray(s.nzval));device_synchronize();t)
cpucopy(s::CudaSparseMatrixCSC)=SparseMatrixCSC(s.m,s.n,to_host(s.colptr),to_host(s.rowval),to_host(s.nzval))
similar(s::CudaSparseMatrixCSC,T,dims::Dims)=gpucopy(spzeros(T,Cint,dims...))

hcat!{T}(x::CudaSparseMatrixCSC{T}, s::CudaSparseMatrixCSC{T},vj,nj)=gpucopy(hcat!(cpucopy(x),cpucopy(s),cpucopy(vj),nj))

# At_mul_B!{T}(k::CudaMatrix{T}, x::CudaSparseMatrixCSC{T}, s::CudaSparseMatrixCSC{T})=A_mul_B!(k,x.',s)

function At_mul_B!(k::CudaMatrix{Float32}, x::CudaSparseMatrixCSC{Float32}, s::CudaSparseMatrixCSC{Float32})
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:At_mul_B_32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    device_synchronize()
    return k
end

function At_mul_B!(k::CudaMatrix{Float64}, x::CudaSparseMatrixCSC{Float64}, s::CudaSparseMatrixCSC{Float64})
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:At_mul_B_64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    device_synchronize()
    return k
end

function A_mul_B!(k::CudaMatrix{Float32}, x::CudaSparseMatrixCSC{Float32}, s::CudaSparseMatrixCSC{Float32})
    @assert size(k)==(size(x,1),size(s,2))
    ccall((:A_mul_B_32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    device_synchronize()
    return k
end

function A_mul_B!(k::CudaMatrix{Float64}, x::CudaSparseMatrixCSC{Float64}, s::CudaSparseMatrixCSC{Float64})
    @assert size(k)==(size(x,1),size(s,2))
    ccall((:A_mul_B_64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    device_synchronize()
    return k
end


transpose(x::CudaSparseMatrixCSC)=(t=gpucopy(cpucopy(x).');device_synchronize();t)

# 100ksv: (128,128) and (512,512) work best for At_test (1.78)
# 10ksv: (9,10), (10,10), (11,8+), (12,7+) (1.44)
function At_test(blk,thr,k::CudaMatrix{Float32}, x::CudaSparseMatrixCSC{Float32}, s::CudaSparseMatrixCSC{Float32})
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:At_test,libkunet),Void,
          (Cint,Cint,Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          blk,thr,size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    device_synchronize()
    return k
end

# 100ksv: (64,128) and *(128,64)* and (512,32) work best for A_test (1.25)
# 10ksv: 1<<(6,8), (7,8), (8,8), (9,8), (10,8) work best for A_test (1.28)
function A_test(blk,thr,k::CudaMatrix{Float32}, x::CudaSparseMatrixCSC{Float32}, s::CudaSparseMatrixCSC{Float32})
    @assert size(k)==(size(x,1),size(s,2))
    ccall((:A_test,libkunet),Void,
          (Cint,Cint,Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          blk,thr,size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
    device_synchronize()
    return k
end

# using CUSPARSE: cusparseHandle, cusparseMatDescrDefault, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE, cusparseDcsr2csc, cusparseScsr2csc

# function transpose1(x::CudaSparseMatrixCSC{Float32})  # this is buggy ???
#     (xrows,xcols) = size(x); nz = nnz(x)
#     y = CudaSparseMatrixCSC(xcols, xrows, CudaArray(zeros(Int32, xrows+1)), CudaArray(zeros(Int32, nz)), CudaArray(zeros(Float32, nz)))
#     cusparseScsr2csc(cusparseHandle, xrows, xcols, nz, x.nzval, x.colptr, x.rowval, y.nzval, y.rowval, y.colptr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE)
#     return y
# end

# using CUSPARSE: cusparseSnnz, cusparseDnnz, cusparseSdense2csc, cusparseDdense2csc, CUSPARSE_DIRECTION_COLUMN, cusparseMatDescrDefault

# function sparse(x::CudaMatrix{Float32})
#     (xrows, xcols) = size(x)
#     nzarray = CudaArray(Int32, xcols)
#     nzcount = Int32[0]
#     cusparseSnnz(cusparseHandle, CUSPARSE_DIRECTION_COLUMN, xrows, xcols, cusparseMatDescrDefault, x, xrows, nzarray, nzcount)
#     nz = int(nzcount[1])
#     y = CudaSparseMatrixCSC(xrows, xcols, CudaArray(Cint, xcols+1), CudaArray(Cint, nz), CudaArray(Float32, nz))
#     cusparseSdense2csc(cusparseHandle, xrows, xcols, cusparseMatDescrDefault, x, xrows, nzarray, y.nzval, y.rowval, y.colptr)
#     return y
# end

# function sparse(x::CudaMatrix{Float64})
#     (xrows, xcols) = size(x)
#     nzarray = CudaArray(Int32, xcols)
#     nzcount = Int32[0]
#     cusparseDnnz(cusparseHandle, CUSPARSE_DIRECTION_COLUMN, xrows, xcols, cusparseMatDescrDefault, x, xrows, nzarray, nzcount)
#     nz = int(nzcount[1])
#     y = CudaSparseMatrixCSC(xrows, xcols, CudaArray(Cint, xcols+1), CudaArray(Cint, nz), CudaArray(Float64, nz))
#     cusparseDdense2csc(cusparseHandle, xrows, xcols, cusparseMatDescrDefault, x, xrows, nzarray, y.nzval, y.rowval, y.colptr)
#     return y
# end

