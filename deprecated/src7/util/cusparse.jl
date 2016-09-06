### fixes

using CUDArt
using CUSPARSE
using CUSPARSE: CudaSparseMatrix

Base.convert(::Type{CudaSparseMatrixCSC}, x::SparseMatrixCSC)=CudaSparseMatrixCSC(x)
Base.convert{T<:Array}(::Type{T},a::CudaSparseMatrix)=full(to_host(a))
Base.isempty(a::CudaSparseMatrix) = (length(a) == 0)
Base.issparse(a::CudaSparseMatrix) = true
# Base.ndims(::CudaSparseMatrix) = 2
Base.nnz(x::CudaSparseMatrix)=x.nnz
Base.scale!(c,a::CudaSparseMatrix) = (scale!(c,a.nzVal); a)
Base.stride(g::CudaSparseMatrix,i)=(i==1 ? 1 : i==2 ? g.dims[1] : length(g))
Base.strides(g::CudaSparseMatrix)=(1,g.dims[1])
# Base.summary(a::CudaSparseMatrix) = string(Base.dims2string(size(a)), " ", typeof(a))
Base.vecnorm(a::CudaSparseMatrix) = vecnorm(a.nzVal)
Base.LinAlg.BLAS.nrm2(a::CudaSparseMatrix) = Base.LinAlg.BLAS.nrm2(a.nzVal)

function copysync!{T}(a::CudaSparseMatrixCSC{T}, b::SparseMatrixCSC{T})
    a.dims = (b.m,b.n)
    a.nnz = convert(Cint, length(b.nzval))
    resizecopy!(a.colPtr, convert(Vector{Cint},b.colptr))
    resizecopy!(a.rowVal, convert(Vector{Cint},b.rowval))
    resizecopy!(a.nzVal, b.nzval)
    gpusync(); return a
end

function copysync!{T}(a::CudaSparseMatrixCSC{T}, b::CudaSparseMatrixCSC{T})
    a.dims = b.dims
    a.nnz = b.nnz
    resizecopy!(a.colPtr, convert(Vector{Cint},b.colPtr))
    resizecopy!(a.rowVal, convert(Vector{Cint},b.rowVal))
    resizecopy!(a.nzVal, b.nzVal)
    gpusync(); return a
end

function copysync!{T}(a::CudaSparseMatrixCSR{T}, b::CudaSparseMatrixCSR{T})
    a.dims = b.dims
    a.nnz = b.nnz
    resizecopy!(a.rowPtr, convert(Vector{Cint},b.rowPtr))
    resizecopy!(a.colVal, convert(Vector{Cint},b.colVal))
    resizecopy!(a.nzVal, b.nzVal)
    gpusync(); return a
end

function resizecopy!{T}(a::CudaVector{T}, b::Vector{T})
    resize!(a, length(b))       # TODO: is this efficient?
    copysync!(a, b)
end

function resizecopy!{T}(a::CudaVector{T}, b::CudaVector{T})
    resize!(a, length(b))       # TODO: is this efficient?
    copysync!(a, b)
end

CUDArt.free(x::CudaSparseMatrixCSR)=(free(x.rowPtr);free(x.colVal);free(x.nzVal))

function fillsync!(x::CudaSparseMatrixCSR,n)
    n == 0 || error("Only 0 fill for sparse")
    fillsync!(x.rowPtr,1)
    resize!(x.colVal,0)
    resize!(x.nzVal,0)
    x.nnz = 0
    return x
end

function fillsync!(x::CudaSparseMatrixCSC,n)
    n == 0 || error("Only 0 fill for sparse")
    fillsync!(x.colPtr,1)
    resize!(x.rowVal,0)
    resize!(x.nzVal,0)
    x.nnz = 0
    return x
end

# A_mul_Bt!(csr,dns,csc) is more efficient if we do not insist on
# unique sorted csr entries.  To signal unsorted csr, we introduce a
# new type so we don't accidentally pass it to an unprepared function.
# The following adapted from CUSPARSE.jl:

# For dw = r' * du we need the CSC versions below

if !isdefined(:CudaSparseMatrixCSRU)
    type CudaSparseMatrixCSRU{T}
        rowPtr::CudaVector{Cint}
        colVal::CudaVector{Cint}
        nzVal::CudaVector{T}
        dims::NTuple{2,Int}
        nnz::Cint
        dev::Int
    end
end

if !isdefined(:CudaSparseMatrixCSCU)
    type CudaSparseMatrixCSCU{T}
        colPtr::CudaVector{Cint}
        rowVal::CudaVector{Cint}
        nzVal::CudaVector{T}
        dims::NTuple{2,Int}
        nnz::Cint
        dev::Int
    end
end

CudaSparseMatrixCSRU(T::Type, m::Integer, n::Integer)=CudaSparseMatrixCSRU{T}(fillsync!(CudaArray(Cint,m+1),1), CudaArray(Cint,0), CudaArray(T,0), (convert(Int,m),convert(Int,n)), convert(Cint,0), convert(Int,device()))
CudaSparseMatrixCSRU(T::Type, rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray, dims::NTuple{2,Int}) = CudaSparseMatrixCSRU{T}(rowPtr, colVal, nzVal, dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSRU(T::Type, rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixCSRU{T}(rowPtr, colVal, nzVal, dims, nnz, device())

CudaSparseMatrixCSCU(T::Type, m::Integer, n::Integer)=CudaSparseMatrixCSCU{T}(fillsync!(CudaArray(Cint,n+1),1), CudaArray(Cint,0), CudaArray(T,0), (convert(Int,m),convert(Int,n)), convert(Cint,0), convert(Int,device()))
CudaSparseMatrixCSCU(T::Type, colPtr::CudaArray, rowVal::CudaArray, nzVal::CudaArray, dims::NTuple{2,Int}) = CudaSparseMatrixCSCU{T}(colPtr, rowVal, nzVal, dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSCU(T::Type, colPtr::CudaArray, rowVal::CudaArray, nzVal::CudaArray, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixCSCU{T}(colPtr, rowVal, nzVal, dims, nnz, device())
CudaSparseMatrixCSCU(T::Type, colPtr::Vector, rowVal::Vector, nzVal::Vector, dims::NTuple{2,Int}) = CudaSparseMatrixCSCU{T}(CudaArray(convert(Vector{Cint},colPtr)), CudaArray(convert(Vector{Cint},rowVal)), CudaArray(nzVal), dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSCU(Mat::SparseMatrixCSC) = CudaSparseMatrixCSCU(eltype(Mat), Mat.colptr, Mat.rowval, Mat.nzval, size(Mat))


Base.eltype{T}(x::CudaSparseMatrixCSRU{T})=T
Base.size(x::CudaSparseMatrixCSRU)=x.dims
Base.issparse(x::CudaSparseMatrixCSRU)=true
Base.scale!(s,x::CudaSparseMatrixCSRU)=scale!(s,x.nzVal)
Base.similar(Mat::CudaSparseMatrixCSRU) = CudaSparseMatrixCSRU(eltype(Mat), copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
Base.copy(Mat::CudaSparseMatrixCSRU; stream=null_stream) = copysync!(similar(Mat),Mat;stream=null_stream)

Base.eltype{T}(x::CudaSparseMatrixCSCU{T})=T
Base.size(x::CudaSparseMatrixCSCU)=x.dims
Base.issparse(x::CudaSparseMatrixCSCU)=true
Base.scale!(s,x::CudaSparseMatrixCSCU)=scale!(s,x.nzVal)
Base.similar(Mat::CudaSparseMatrixCSCU) = CudaSparseMatrixCSCU(eltype(Mat), copy(Mat.colPtr), copy(Mat.rowVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
Base.copy(Mat::CudaSparseMatrixCSCU; stream=null_stream) = copysync!(similar(Mat),Mat;stream=null_stream)

function copysync!(dst::CudaSparseMatrixCSRU, src::CudaSparseMatrixCSRU; stream=null_stream)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    resizecopy!( dst.rowPtr, src.rowPtr )
    resizecopy!( dst.colVal, src.colVal )
    resizecopy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    gpusync(); return dst
end

function copysync!(dst::CudaSparseMatrixCSCU, src::CudaSparseMatrixCSCU; stream=null_stream)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    resizecopy!( dst.colPtr, src.colPtr )
    resizecopy!( dst.rowVal, src.rowVal )
    resizecopy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    gpusync(); return dst
end

function CUDArt.to_host{T}(Mat::CudaSparseMatrixCSRU{T})
    rowPtr = to_host(Mat.rowPtr)
    colVal = to_host(Mat.colVal)
    nzVal = to_host(Mat.nzVal)
    #construct Is
    I = similar(colVal)
    counter = 1
    for row = 1 : size(Mat)[1], k = rowPtr[row] : (rowPtr[row+1]-1)
        I[counter] = row
        counter += 1
    end
    return sparse(I,colVal,nzVal,Mat.dims[1],Mat.dims[2])
end

function CUDArt.to_host{T}(Mat::CudaSparseMatrixCSCU{T})
    (m,n) = Mat.dims
    colptr = to_host(Mat.colPtr)
    rowval = to_host(Mat.rowVal)
    nzval = to_host(Mat.nzVal)
    colval = similar(rowval)
    for i=1:n
        colval[colptr[i]:colptr[i+1]-1] = i
    end
    return sparse(rowval, colval, nzval, m, n)
end

deepcopy_internal(x::CudaSparseMatrixCSRU, s::ObjectIdDict)=(haskey(s,x)||(s[x]=copy(x));s[x])
gpucopy_internal(x::CudaSparseMatrixCSRU, s::ObjectIdDict)=deepcopy_internal(x,s)

deepcopy_internal(x::CudaSparseMatrixCSCU, s::ObjectIdDict)=(haskey(s,x)||(s[x]=copy(x));s[x])
gpucopy_internal(x::CudaSparseMatrixCSCU, s::ObjectIdDict)=deepcopy_internal(x,s)

deepcopy_internal(x::CudaSparseMatrix, s::ObjectIdDict)=(haskey(s,x)||(s[x]=copy(x));s[x])
gpucopy_internal(x::CudaSparseMatrix, s::ObjectIdDict)=deepcopy_internal(x,s)
gpucopy_internal{T<:Number}(x::SparseMatrixCSC{T}, s::ObjectIdDict)=(haskey(s,x)||(s[x]=CudaSparseMatrixCSC(x));s[x])

# We need cpu versions of CudaSparseMatrices for cpu/gpucopy to work
if !isdefined(:SparseMatrixCSR0)
    type SparseMatrixCSR0{T}
        rowPtr::Array{Cint,1}
        colVal::Array{Cint,1}
        nzVal::Array{T,1}
        dims::NTuple{2,Int}
        nnz::Cint
        dev::Int
    end
end

if !isdefined(:SparseMatrixCSRU)
    type SparseMatrixCSRU{T}
        rowPtr::Array{Cint,1}
        colVal::Array{Cint,1}
        nzVal::Array{T,1}
        dims::NTuple{2,Int}
        nnz::Cint
        dev::Int
    end
end

if !isdefined(:SparseMatrixCSC0)
    type SparseMatrixCSC0{T}
        colPtr::Array{Cint,1}
        rowVal::Array{Cint,1}
        nzVal::Array{T,1}
        dims::NTuple{2,Int}
        nnz::Cint
        dev::Int
    end
end

if !isdefined(:SparseMatrixCSCU)
    type SparseMatrixCSCU{T}
        colPtr::Array{Cint,1}
        rowVal::Array{Cint,1}
        nzVal::Array{T,1}
        dims::NTuple{2,Int}
        nnz::Cint
        dev::Int
    end
end

# preserve matrix format in cpu
isdefined(:SparseMatrix0) || (typealias SparseMatrix0 Union{SparseMatrixCSC0,SparseMatrixCSR0,SparseMatrixCSCU,SparseMatrixCSRU})
isdefined(:CudaSparseMatrix0) || typealias CudaSparseMatrix0 Union{CudaSparseMatrixCSC,CudaSparseMatrixCSR,CudaSparseMatrixCSCU,CudaSparseMatrixCSRU}

cpucopy_internal(x::CudaSparseMatrix0, s::ObjectIdDict)=(haskey(s,x)||(s[x]=cpucopy_sparse(x));s[x])
gpucopy_internal(x::SparseMatrix0, s::ObjectIdDict)=(haskey(s,x)||(s[x]=gpucopy_sparse(x));s[x])

cpucopy_sparse{T}(x::CudaSparseMatrixCSC{T})=SparseMatrixCSC0{T}(to_host(x.colPtr),to_host(x.rowVal),to_host(x.nzVal),x.dims,x.nnz,x.dev)
cpucopy_sparse{T}(x::CudaSparseMatrixCSCU{T})=SparseMatrixCSCU{T}(to_host(x.colPtr),to_host(x.rowVal),to_host(x.nzVal),x.dims,x.nnz,x.dev)
cpucopy_sparse{T}(x::CudaSparseMatrixCSR{T})=SparseMatrixCSR0{T}(to_host(x.rowPtr),to_host(x.colVal),to_host(x.nzVal),x.dims,x.nnz,x.dev)
cpucopy_sparse{T}(x::CudaSparseMatrixCSRU{T})=SparseMatrixCSRU{T}(to_host(x.rowPtr),to_host(x.colVal),to_host(x.nzVal),x.dims,x.nnz,x.dev)

gpucopy_sparse{T}(x::SparseMatrixCSC0{T})=CudaSparseMatrixCSC{T}(CudaArray(x.colPtr),CudaArray(x.rowVal),CudaArray(x.nzVal),x.dims,x.nnz,x.dev)
gpucopy_sparse{T}(x::SparseMatrixCSCU{T})=CudaSparseMatrixCSCU{T}(CudaArray(x.colPtr),CudaArray(x.rowVal),CudaArray(x.nzVal),x.dims,x.nnz,x.dev)
gpucopy_sparse{T}(x::SparseMatrixCSR0{T})=CudaSparseMatrixCSR{T}(CudaArray(x.rowPtr),CudaArray(x.colVal),CudaArray(x.nzVal),x.dims,x.nnz,x.dev)
gpucopy_sparse{T}(x::SparseMatrixCSRU{T})=CudaSparseMatrixCSRU{T}(CudaArray(x.rowPtr),CudaArray(x.colVal),CudaArray(x.nzVal),x.dims,x.nnz,x.dev)

# equality testing
function Base.isequal(a::Union{CudaSparseMatrix0,SparseMatrix0},b::Union{CudaSparseMatrix0,SparseMatrix0})
    typeof(a)==typeof(b) || return false
    for n in fieldnames(a)
        if isdefined(a,n) && isdefined(b,n)
            isequal(a.(n), b.(n)) || return false
        elseif isdefined(a,n) || isdefined(b,n)
            return false
        end
    end
    return true
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
            statuscheck(ccall((:cusparseXcsrgemmNnz,libcusparse), cusparseStatus_t, # t:186/868
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
            statuscheck(ccall(($(string(fname)),libcusparse), cusparseStatus_t, # t:659/868
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
# # This is buggy in master:util.jl:75, submitted pull request:
# function to_host2{T}(Mat::CudaSparseMatrixCSR{T})
#     rowPtr = to_host(Mat.rowPtr)
#     colVal = to_host(Mat.colVal)
#     nzVal = to_host(Mat.nzVal)
#     #construct Is
#     I = similar(colVal)
#     counter = 1
#     for row = 1 : size(Mat)[1], k = rowPtr[row] : (rowPtr[row+1]-1)
#         I[counter] = row
#         counter += 1
#     end
#     return sparse(I,colVal,nzVal,Mat.dims[1],Mat.dims[2])
# end

