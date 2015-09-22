### fixes

using CUDArt
using CUSPARSE

Base.convert(::Type{CudaSparseMatrixCSC}, x::SparseMatrixCSC)=CudaSparseMatrixCSC(x)
Base.convert{T<:Array}(::Type{T},a::CudaSparseMatrix)=full(to_host(a))
Base.isempty(a::CudaSparseMatrix) = (length(a) == 0)
Base.issparse(a::CudaSparseMatrix) = true
Base.ndims(::CudaSparseMatrix) = 2
Base.nnz(x::CudaSparseMatrix)=x.nnz
Base.scale!(c,a::CudaSparseMatrix) = (scale!(c,a.nzVal); a)
Base.stride(g::CudaSparseMatrix,i)=(i==1 ? 1 : i==2 ? g.dims[1] : length(g))
Base.strides(g::CudaSparseMatrix)=(1,g.dims[1])
Base.summary(a::CudaSparseMatrix) = string(Base.dims2string(size(a)), " ", typeof(a))
Base.vecnorm(a::CudaSparseMatrix) = vecnorm(a.nzVal)

function Base.copy!{T}(a::CudaSparseMatrixCSC{T}, b::SparseMatrixCSC{T})
    a.dims = (b.m,b.n)
    a.nnz = convert(Cint, length(b.nzval))
    resizecopy!(a.colPtr, convert(Vector{Cint},b.colptr))
    resizecopy!(a.rowVal, convert(Vector{Cint},b.rowval))
    resizecopy!(a.nzVal, b.nzval)
    return a
end

function resizecopy!{T}(a::CudaVector{T}, b::Vector{T})
    resize!(a, length(b))       # TODO: is this efficient?
    copy!(a, b)
end

import Base.LinAlg.BLAS: gemm!
using CUSPARSE: SparseChar, cusparseop, cusparseindex,
    cusparseMatDescr_t, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT,
    statuscheck, libcusparse, cusparseStatus_t, cusparseHandle_t, cusparseOperation_t, cusparsehandle

for (fname,elty) in ((:cusparseScsrgemm, :Float32),
                     (:cusparseDcsrgemm, :Float64),
                     (:cusparseCcsrgemm, :Complex64),
                     (:cusparseZcsrgemm, :Complex128))
    @eval begin
        function gemm!(transa::SparseChar,
                       transb::SparseChar,
                       A::CudaSparseMatrixCSR{$elty},
                       B::CudaSparseMatrixCSR{$elty},
                       C::CudaSparseMatrixCSR{$elty})
            cutransa = cusparseop(transb)
            cutransb = cusparseop(transa)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cusparseindex('O'))
            m,k  = transa == 'N' ? A.dims : (A.dims[2],A.dims[1])
            kB,n = transb == 'N' ? B.dims : (B.dims[2],B.dims[1])
            if k != kB
                throw(DimensionMismatch("Interior dimension of A, $k, and B, $kB, must match"))
            end
            nnzC = Array(Cint,1)
            resize!(C.rowPtr, m+1)
            statuscheck(ccall((:cusparseXcsrgemmNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
                               Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], cutransa, cutransb,
                              m, n, k, &cudesc, A.nnz, A.rowPtr, A.colVal,
                              &cudesc, B.nnz, B.rowPtr, B.colVal, &cudesc,
                              C.rowPtr, nnzC))
            C.nnz = nnzC[1]
            C.dims = (m,n)
            resize!(C.nzVal, C.nnz)
            resize!(C.colVal, C.nnz)
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, Ptr{$elty},
                               Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t},
                               Cint, Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}), cusparsehandle[1], cutransa,
                              cutransb, m, n, k, &cudesc, A.nnz, A.nzVal,
                              A.rowPtr, A.colVal, &cudesc, B.nnz, B.nzVal,
                              B.rowPtr, B.colVal, &cudesc, C.nzVal,
                              C.rowPtr, C.colVal))
            C
        end
    end
end


        # function geam!(alpha::$elty,
        #                A::CudaSparseMatrixCSR{$elty},
        #                beta::$elty,
        #                B::CudaSparseMatrixCSR{$elty},
        #                C::CudaSparseMatrixCSR{$elty},
        #                indexA::SparseChar='O',
        #                indexB::SparseChar='O',
        #                indexC::SparseChar='O')
        #     cuinda = cusparseindex(indexA)
        #     cuindb = cusparseindex(indexB)
        #     cuindc = cusparseindex(indexB)
        #     cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
        #     cudescb = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
        #     cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
        #     mA,nA = A.dims
        #     mB,nB = B.dims
        #     if ( (mA != mB) || (nA != nB) )
        #         throw(DimensionMismatch(""))
        #     end
        #     nnzC = Array(Cint,1)
        #     length(C.rowPtr) < mA+1 && (C.rowPtr = CudaArray(zeros(Cint, mA+1)))
        #     statuscheck(ccall((:cusparseXcsrgeamNnz,libcusparse), cusparseStatus_t,
        #                       (cusparseHandle_t, Cint, Cint,
        #                        Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
        #                        Ptr{Cint}, Ptr{cusparseMatDescr_t}, Cint, Ptr{Cint},
        #                        Ptr{Cint}, Ptr{cusparseMatDescr_t}, Ptr{Cint},
        #                        Ptr{Cint}), cusparsehandle[1], mA, nA, &cudesca,
        #                       A.nnz, A.rowPtr, A.colVal, &cudescb, B.nnz,
        #                       B.rowPtr, B.colVal, &cudescc, C.rowPtr, nnzC))
        #     C.nnz = nnzC[1]
        #     C.dims = A.dims
        #     length(C.nzVal)  < C.nnz && (C.nzVal = CudaArray($elty,Int(C.nnz)))
        #     length(C.colVal) < C.nnz && (C.colVal = CudaArray(Cint,Int(C.nnz)))
        #     statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t,
        #                       (cusparseHandle_t, Cint, Cint, Ptr{$elty},
        #                        Ptr{cusparseMatDescr_t}, Cint, Ptr{$elty},
        #                        Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
        #                        Ptr{cusparseMatDescr_t}, Cint, Ptr{$elty},
        #                        Ptr{Cint}, Ptr{Cint}, Ptr{cusparseMatDescr_t},
        #                        Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
        #                       cusparsehandle[1], mA, nA, [alpha], &cudesca,
        #                       A.nnz, A.nzVal, A.rowPtr, A.colVal, [beta],
        #                       &cudescb, B.nnz, B.nzVal, B.rowPtr, B.colVal,
        #                       &cudescc, C.nzVal, C.rowPtr, C.colVal))
        #     C
        # end
