# TODO: deprecate KUdense in this file.

using Base.LinAlg.BLAS: gemm!, scal!, nrm2
import Base: A_mul_B!, A_mul_Bt!, At_mul_B!, vecnorm, sum
import Base.LinAlg: axpy!, scale!

### VEC functions
axpy!{T}(a::Number,x::CudaArray{T},y::CudaArray{T})=(n=length(x); n==length(y)||error(); axpy!(n,T(a),x,1,y,1); gpusync(); y)
scale!{T}(a::Number,x::CudaArray{T})=(a==1||scal!(length(x),T(a),x,1); gpusync(); x)
scale!{T}(x::CudaArray{T},a::Number)=(a==1||scal!(length(x),T(a),x,1); gpusync(); x)

# CUBLAS is twice as slow as Barret's custom kernel in my experiments:
# vecnorm(x::CudaArray)=nrm2(x)
vecnorm2(x::CudaArray{Float32})=ccall((:vecnorm2_32,libknet),Float32,(Ptr{Cfloat},Cint),x,length(x))
vecnorm2(x::CudaArray{Float64})=ccall((:vecnorm2_64,libknet),Float64,(Ptr{Cdouble},Cint),x,length(x))
vecnorm1(x::CudaArray{Float32})=ccall((:vecnorm1_32,libknet),Float32,(Ptr{Cfloat},Cint),x,length(x))
vecnorm1(x::CudaArray{Float64})=ccall((:vecnorm1_64,libknet),Float64,(Ptr{Cdouble},Cint),x,length(x))
sum(x::CudaArray{Float32})=ccall((:sum32,libknet),Float32,(Ptr{Cfloat},Cint),x,length(x))
sum(x::CudaArray{Float64})=ccall((:sum64,libknet),Float64,(Ptr{Cdouble},Cint),x,length(x))
vecnorm(x::CudaArray,p=2)=(p==2 ? vecnorm2(x) : p==1 ? vecnorm1(x) : error("Undefined"))

# (ccall((:axpy32csr,libknet),Void,(Cint,Cint,Cfloat,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),x.dims[1],x.dims[2],convert(Float32,a),x.nnz,x.nzVal,x.rowPtr,x.colVal,y); y)

### MMUL
# This is not a complete implementation.  The goal is to support Knet
# operations for sparse/dense matrices on cpu/gpu.  The operations needed:
#
# mmul forw: A_mul_B!(y, w, x)		A_mul_Bs!(y, w, x): cpu/gpu: kudense, array, sparse
# mmul back: A_mul_Bt!(dw, dy, x)	A_mul_Bst!(dw, dy, x): cpu/gpu: array, kudense, sparse
# mmul back: At_mul_B!(dx, w, dy)	no dx: only initial input can be sparse
# kper forw: At_mul_B!(k, s, x)		Ast_mul_Bs!(k, s, x): cpu/gpu: kudense, kusparse, sparse


### CudaMatrix (Matrix versions already defined)

A_mul_B!{T}( C::CudaMatrix{T}, A::CudaMatrix{T}, B::CudaMatrix{T})=(gemm!('N','N',one(T),A,B,zero(T),C); gpusync(); C)
A_mul_Bt!{T}(C::CudaMatrix{T}, A::CudaMatrix{T}, B::CudaMatrix{T})=(gemm!('N','T',one(T),A,B,zero(T),C); gpusync(); C)
At_mul_B!{T}(C::CudaMatrix{T}, A::CudaMatrix{T}, B::CudaMatrix{T})=(gemm!('T','N',one(T),A,B,zero(T),C); gpusync(); C)

### Add the ability to multiply arrays with other than 2 dimensions
A_mul_B!{T}(C::CudaArray{T},A::CudaArray{T},B::CudaArray{T})=(gemm!('N','N',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); gpusync(); C)
A_mul_Bt!{T}(C::CudaArray{T},A::CudaArray{T},B::CudaArray{T})=(gemm!('N','T',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); gpusync(); C)
At_mul_B!{T}(C::CudaArray{T},A::CudaArray{T},B::CudaArray{T})=(gemm!('T','N',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); gpusync(); C)

# CudaSparseMatrixCSC
# y = w * xS
function A_mul_B!{T}(C::CudaMatrix{T}, A::CudaMatrix{T}, B::CudaSparseMatrixCSC{T})
    # cusparse only supports mul with csr x dense.
    bT = CudaSparseMatrixCSR{T}(B.colPtr, B.rowVal, B.nzVal, (B.dims[2],B.dims[1]), B.nnz, B.dev)
    cT = similar(C, reverse(size(C)))
    # TODO: avoid alloc by using inplace:
    # cT = CudaArray{T,2}(C.ptr, reverse(size(C)), C.dev)
    CUSPARSE.csrmm2!('N','T',one(T),bT,A,zero(T),cT,'O') # yT = xT * w'
    CUBLAS.geam!('T','T',one(T),cT,zero(T),cT,C)
    free(cT)
    gpusync()
    return C
end

# CudaSparseMatrixCSR
# u = rS * w
function A_mul_B!{T}(C::CudaArray{T}, A::CudaSparseMatrixCSR{T}, B::CudaArray{T})
    CUSPARSE.csrmm!('N',one(T),mat2d(A),mat2d(B),zero(T),C,'O')
end

# dw = dy * x'
function A_mul_Bt!{T}(C::CudaSparseMatrixCSRU{T},A::CudaMatrix{T},B::CudaSparseMatrixCSC{T})
    C.dims == (size(A,1), size(B,1)) || error()
    C.nnz = C.dims[1] * B.nnz
    resize!(C.rowPtr, C.dims[1]+1)
    resize!(C.colVal, C.nnz)
    resize!(C.nzVal, C.nnz)
    # Treating B as CSR effectively transposes it.
    T <: Float32 ? ccall((:mul_dns_csr_csru_32,libknet), Void,
                         (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
                         size(A,1), size(A,2), A, B.colPtr, B.rowVal, B.nzVal, C.rowPtr, C.colVal, C.nzVal) :
    T <: Float64 ? ccall((:mul_dns_csr_csru_64,libknet), Void,
                         (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
                         size(A,1), size(A,2), A, B.colPtr, B.rowVal, B.nzVal, C.rowPtr, C.colVal, C.nzVal) :
    error("$T not supported")
    gpusync()
    return C
end

function At_mul_B!{T}(C::CudaSparseMatrixCSCU{T},A::CudaSparseMatrixCSR{T},B::CudaMatrix{T})
    C.dims == (size(A,2), size(B,2)) || error()
    C.nnz = C.dims[2] * A.nnz
    resize!(C.colPtr, C.dims[2]+1)
    resize!(C.rowVal, C.nnz)
    resize!(C.nzVal, C.nnz)
    T <: Float32 ? ccall((:mul_csc_dns_cscu_32,libknet), Void,
                         (Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
                         size(B,1), size(B,2), A.rowPtr, A.colVal, A.nzVal, B, C.colPtr, C.rowVal, C.nzVal) :
    T <: Float64 ? ccall((:mul_csc_dns_cscu_64,libknet), Void,
                         (Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
                         size(B,1), size(B,2), A.rowPtr, A.colVal, A.nzVal, B, C.colPtr, C.rowVal, C.nzVal) :
    error("$T not supported")
    gpusync()
    return C
end

# dw = dy * x'
# This version gives a proper CSR matrix but is slower:
function A_mul_Bt!{T}(C::CudaSparseMatrixCSR{T},A::CudaMatrix{T},B::CudaSparseMatrixCSC{T})
    bT = CudaSparseMatrixCSR{T}(B.colPtr, B.rowVal, B.nzVal, (B.dims[2],B.dims[1]), B.nnz, B.dev)
    gpusync()
    a = sparse(A)               # t:337 gives CudaSparseMatrixCSR
    gpusync()
    gemm!('N','N',a,bT,C)       # t:868
    gpusync()
    free(a)                     # t:96
    gpusync()
    return C
end

# dw = dw + iw
# This is also slow, so instead we keep dw dense, 
# or use an accumulator type defined below
function axpy!{T}(a,x::CudaSparseMatrixCSR{T},y::CudaSparseMatrixCSR{T})
    z = CUSPARSE.geam(convert(T,a), x, one(T), y, 'O', 'O', 'O')
    free(y)
    y.rowPtr = z.rowPtr
    y.colVal = z.colVal
    y.nzVal = z.nzVal
    y.nnz = z.nnz
    gpusync()
    return y
end

# For incremental updates with sparse dw, it is inefficient to try dw
# += iw.  Instead we will just keep a list of iw.  The sparsity
# patterns are likely different, so we are not losing any space.  So
# we invent a new type, so the user can pretend this is a matrix.

# Unfortunately this is not faster than keeping dw dense when iw sparse.
# However it takes much less space.

type ArrayAccumulator; arr; eltype; size; cnt;
    ArrayAccumulator(xtype, dims)=new(Any[], xtype, dims, 0)
end

Base.size(x::ArrayAccumulator)=x.size
Base.eltype(x::ArrayAccumulator)=x.eltype
fillsync!(x::ArrayAccumulator, n)=(n==0 ? x.cnt = 0 : error())
Base.scale!(s, x::ArrayAccumulator)=(for i=1:x.cnt; scale!(s,x.arr[i]); end; x)

# TODO: this does not work for unsorted csru or ArrayAccumulator,
# since they may have multiple entries for a single position.  Note
# that p=1 is not feasible either since entries might cancel out.
# Base.vecnorm(x::ArrayAccumulator)=(n=0;for i=1:x.cnt;
# n+=vecnorm(x.arr[i]); end; n)
vecnorm(x::ArrayAccumulator,p=2)=(Base.warn_once("Cannot compute vecnorm for $(typeof(x)), returning 0");0)

function axpy!(a, x, y::ArrayAccumulator)
    @assert size(x)==size(y) && eltype(x)==eltype(y)
    y.cnt += 1
    if length(y.arr) >= y.cnt
        copysync!(y.arr[y.cnt], x)
    elseif length(y.arr)+1 == y.cnt
        push!(y.arr, copy(x))
    else
        error()
    end
    a != 1 && scale!(a, y.arr[y.cnt])
    gpusync()
    return y
end

function axpy!(a, x::ArrayAccumulator, y::CudaMatrix)
    @assert size(x)==size(y) && eltype(x)==eltype(y)
    for i=1:x.cnt
        axpy!(a, x.arr[i], y)
    end
    gpusync()
    return y
end


# CSR does not need atomic operations because each position is only written once
axpy!(a,x::CudaSparseMatrixCSR{Float32},y::CudaMatrix{Float32})=(ccall((:add_csr_dns_32,libknet),Void,(Cint,Cint,Cfloat,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),x.dims[1],x.dims[2],convert(Float32,a),x.nnz,x.nzVal,x.rowPtr,x.colVal,y); gpusync(); y)
axpy!(a,x::CudaSparseMatrixCSR{Float64},y::CudaMatrix{Float64})=(ccall((:add_csr_dns_64,libknet),Void,(Cint,Cint,Cdouble,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),x.dims[1],x.dims[2],convert(Float64,a),x.nnz,x.nzVal,x.rowPtr,x.colVal,y); gpusync(); y)
# CSRU needs atomic operations because the sparse representation may contain multiple values for one location
axpy!(a,x::CudaSparseMatrixCSRU{Float32},y::CudaMatrix{Float32})=(ccall((:add_csr_dns_atomic_32,libknet),Void,(Cint,Cint,Cfloat,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),x.dims[1],x.dims[2],convert(Float32,a),x.nnz,x.nzVal,x.rowPtr,x.colVal,y); gpusync(); y)
axpy!(a,x::CudaSparseMatrixCSRU{Float64},y::CudaMatrix{Float64})=(ccall((:add_csr_dns_atomic_64,libknet),Void,(Cint,Cint,Cdouble,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),x.dims[1],x.dims[2],convert(Float64,a),x.nnz,x.nzVal,x.rowPtr,x.colVal,y); gpusync(); y)

# CSC does not need atomic operations because each position is only written once
axpy!(a,x::CudaSparseMatrixCSC{Float32},y::CudaMatrix{Float32})=(ccall((:add_csc_dns_32,libknet),Void,(Cint,Cint,Cfloat,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),x.dims[1],x.dims[2],convert(Float32,a),x.nnz,x.nzVal,x.colPtr,x.rowVal,y); gpusync(); y)
axpy!(a,x::CudaSparseMatrixCSC{Float64},y::CudaMatrix{Float64})=(ccall((:add_csc_dns_64,libknet),Void,(Cint,Cint,Cdouble,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),x.dims[1],x.dims[2],convert(Float64,a),x.nnz,x.nzVal,x.colPtr,x.rowVal,y); gpusync(); y)
# CSCU needs atomic operations because the sparse representation may contain multiple values for one location
axpy!(a,x::CudaSparseMatrixCSCU{Float32},y::CudaMatrix{Float32})=(ccall((:add_csc_dns_atomic_32,libknet),Void,(Cint,Cint,Cfloat,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),x.dims[1],x.dims[2],convert(Float32,a),x.nnz,x.nzVal,x.colPtr,x.rowVal,y); gpusync(); y)
axpy!(a,x::CudaSparseMatrixCSCU{Float64},y::CudaMatrix{Float64})=(ccall((:add_csc_dns_atomic_64,libknet),Void,(Cint,Cint,Cdouble,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),x.dims[1],x.dims[2],convert(Float64,a),x.nnz,x.nzVal,x.colPtr,x.rowVal,y); gpusync(); y)

# This is necessary for cpu ygold in softloss
axpy!{T}(a,x::SparseMatrixCSC{T},y::CudaMatrix{T})=axpy!(a,CudaSparseMatrixCSC(x),y)
axpy!{T}(a,x::Matrix{T},y::CudaMatrix{T})=axpy!(a,CudaArray(x),y)

# Warn we cannot compute vecnorm
vecnorm(x::CudaSparseMatrixCSRU,p=2)=(Base.warn_once("Cannot compute vecnorm for $(typeof(x)), returning 0");0)
vecnorm(x::CudaSparseMatrixCSCU,p=2)=(Base.warn_once("Cannot compute vecnorm for $(typeof(x)), returning 0");0)

### element-wise log and exp:
log!(a::Array,b::Array=a)=(@assert length(a)==length(b); for i=1:length(a); b[i]=log(a[i]); end; b)
exp!(a::Array,b::Array=a)=(@assert length(a)==length(b); for i=1:length(a); b[i]=exp(a[i]); end; b)

Base.log(a::CudaArray)=log!(a,similar(a))
Base.exp(a::CudaArray)=exp!(a,similar(a))

function log!{T}(a::CudaArray{T},b::CudaArray{T}=a)
    length(a)==length(b) || throw(DimensionMismatch())
    T <: Float32 ? ccall((:log32,libknet),Void,(Cint,Ptr{Cfloat},Ptr{Cfloat}),length(a),a,b) :
    T <: Float64 ? ccall((:log64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble}),length(a),a,b) :
    error("$T not supported")
    gpusync()
    return b
end

function exp!{T}(a::CudaArray{T},b::CudaArray{T}=a)
    length(a)==length(b) || throw(DimensionMismatch())
    T <: Float32 ? ccall((:exp32,libknet),Void,(Cint,Ptr{Cfloat},Ptr{Cfloat}),length(a),a,b) :
    T <: Float64 ? ccall((:exp64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble}),length(a),a,b) :
    error("$T not supported")
    gpusync()
    return b
end

### get the diagonal of a matrix:
function diag!{T}(a::CudaMatrix{T}, d::CudaVector{T})
    n = min(size(a)...)
    T <: Float32 ? ccall((:diag32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat}),size(a,1),size(a,2),a,d) :
    T <: Float64 ? ccall((:diag64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble}),size(a,1),size(a,2),a,d) :
    error("$T not supported")
    gpusync()
    return d
end

diag(a::CudaMatrix)=diag!(a, similar(a,(min(size(a)...),)))

function diag!(a::Matrix, d::Vector)
    min(size(a)...) == length(d) || throw(DimensionMismatch())
    for i=1:length(d); d[i] = a[i,i]; end
    return d
end

function diagm!(d::Vector, a::Matrix)
    min(size(a)...) == length(d) || throw(DimensionMismatch())
    fillsync!(a,0)
    for i=1:length(d); a[i,i]=d[i]; end
    return a
end

diagm(d::CudaVector)=diagm!(d, similar(d, (length(d), length(d))))

function diagm!{T}(d::CudaVector{T}, a::CudaMatrix{T})
    min(size(a)...) == length(d) || throw(DimensionMismatch())
    fillsync!(a,0)
    T <: Float32 ? ccall((:diagm32,libknet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat}),size(a,1),size(a,2),d,a) :
    T <: Float64 ? ccall((:diagm64,libknet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble}),size(a,1),size(a,2),d,a) :
    error("$T not supported")
    gpusync()
    return a
end

### DEAD CODE:

# ### axpb! useful scale and shift transformation: x -> ax+b

# axpb!(a::Number, b::Number, x::Array)=(for i=1:length(x); x[i]=a*x[i]+b; end; x)
# axpb!(a::Number, b::Number, x::CudaArray{Float32})=(ccall((:axpb32,libknet),Void,(Cint,Cfloat,Cfloat,Ptr{Cfloat}),length(x),a,b,x); gpusync(); x)
# axpb!(a::Number, b::Number, x::CudaArray{Float64})=(ccall((:axpb64,libknet),Void,(Cint,Cdouble,Cdouble,Ptr{Cdouble}),length(x),a,b,x); gpusync(); x)


### SPARSE: A_mul_Bs!(y, w, x)

# A_mul_B!{A<:Array,B<:Array}(y::KUdense{A}, w::Array, x::KUsparse{B})=(A_mul_B!(y.arr, w, x); y)
# A_mul_B!{A<:CudaArray,B<:CudaArray}(y::KUdense{A}, w::CudaArray, x::KUsparse{B})=(A_mul_B!(y.arr, w, x); y)

# function A_mul_B!{A<:Array}(y::Matrix, w::Matrix, x::KUsparse{A}) # 1607
#     @assert size(y)==(size(w,1), size(x,2))
#     # eltype's do not have to match.
#     fillsync!(y, zero(eltype(y)))
#     @inbounds for xcol=1:size(x,2)
#         @inbounds for xp=x.colptr[xcol]:(x.colptr[xcol+1]-1)
#             xval = x.nzval[xp]  # 133
#             xrow = x.rowval[xp] # wcol
#             @inbounds for wrow=1:size(w,1)
#                 wval = w[wrow,xrow]
#                 yinc = wval * xval  # 245
#                 y[wrow,xcol] += yinc # 789
#             end
#         end
#     end
#     return y
# end

# function A_mul_B!{A<:CudaArray}(y::CudaArray{Float32,2}, w::CudaArray{Float32,2}, x::KUsparse{A,Float32})
#     @assert size(y)==(size(w,1),size(x,2))
#     ccall((:A_mul_Bs_32,libknet),Void,
#           (Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
#           size(w,1),size(x,2),w,x.nzval,x.rowval,x.colptr,y)
#     return y
# end

# function A_mul_B!{A<:CudaArray}(y::CudaArray{Float64,2}, w::CudaArray{Float64,2}, x::KUsparse{A,Float64})
#     @assert size(y)==(size(w,1),size(x,2))
#     ccall((:A_mul_Bs_64,libknet),Void,
#           (Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
#           size(w,1),size(x,2),w,x.nzval,x.rowval,x.colptr,y)
#     return y
# end

### SPARSE: A_mul_Bst!(dw, dy, x)

# A_mul_Bt!{A<:Array,B<:Array}(dw::Array, dy::KUdense{A}, x::KUsparse{B})=A_mul_Bt!(dw, dy.arr, x)
# A_mul_Bt!{A<:CudaArray,B<:CudaArray}(dw::CudaArray, dy::KUdense{A}, x::KUsparse{B})=A_mul_Bt!(dw, dy.arr, x)

# function A_mul_Bt!{A<:Array}(dw::Matrix, dy::Matrix, x::KUsparse{A})
#     @assert size(dw)==(size(dy,1), size(x,1))
#     fillsync!(dw, zero(eltype(dw)))
#     @inbounds for xcol=1:size(x,2)                      # xcol = ycol
#         xrange = x.colptr[xcol]:(x.colptr[xcol+1]-1)	
#         @inbounds for xp in xrange
#             xrow = x.rowval[xp]                         # xrow = wcol
#             xval = x.nzval[xp]
#             @inbounds for yrow=1:size(dy,1)             # yrow = wrow
#                 yval = dy[yrow,xcol]
#                 winc = xval * yval
#                 dw[yrow,xrow] += winc
#             end
#         end
#     end
#     return dw
# end

# function A_mul_Bt!{A<:CudaArray}(dw::CudaArray{Float32,2}, dy::CudaArray{Float32,2}, x::KUsparse{A,Float32})
#     @assert size(dw)==(size(dy,1),size(x,1))
#     ccall((:A_mul_Bst_32,libknet),Void,
#           (Cint,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
#           size(dy,1),size(dy,2),size(x,1),dy,x.nzval,x.rowval,x.colptr,dw)
#     return dw
# end

# function A_mul_Bt!{A<:CudaArray}(dw::CudaArray{Float64,2}, dy::CudaArray{Float64,2}, x::KUsparse{A,Float64})
#     @assert size(dw)==(size(dy,1),size(x,1))
#     ccall((:A_mul_Bst_64,libknet),Void,
#           (Cint,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
#           size(dy,1),size(dy,2),size(x,1),dy,x.nzval,x.rowval,x.colptr,dw)
#     return dw
# end


### SPARSE: Ast_mul_Bs!(k, s, x)

# At_mul_B!{A}(k::KUdense{A}, s::KUsparse, x::KUsparse)=(At_mul_B!(k.arr, s, x); k)

# At_mul_B!{A<:Array}(k::Matrix, s::KUsparse{A}, x::KUsparse{A})=At_mul_B!(k, convert(SparseMatrixCSC,s), convert(SparseMatrixCSC,x))

# At_mul_B!(k::Matrix, s::SparseMatrixCSC, x::SparseMatrixCSC)=copysync!(k, s' * x)

# function At_mul_B!{A<:CudaArray,B<:CudaArray}(k::CudaArray{Float32,2}, s::KUsparse{A,Float32}, x::KUsparse{B,Float32})
#     @assert size(k)==(size(s,2),size(x,2))
#     ccall((:Ast_mul_Bs_32,libknet),Void,
#           (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
#           size(s,2),size(x,2),s.nzval,s.rowval,s.colptr,x.nzval,x.rowval,x.colptr,k)
#     return k
# end

# function At_mul_B!{A<:CudaArray,B<:CudaArray}(k::CudaArray{Float64,2}, s::KUsparse{A,Float64}, x::KUsparse{B,Float64})
#     @assert size(k)==(size(s,2),size(x,2))
#     ccall((:Ast_mul_Bs_64,libknet),Void,
#           (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
#           size(s,2),size(x,2),s.nzval,s.rowval,s.colptr,x.nzval,x.rowval,x.colptr,k)
#     return k
# end



### KUSPARSE

# A_mul_B!{A}(k::KUdense{A}, x::KUsparse{A}, s::KUsparse{A})=
#     (A_mul_B!(convert(A, k), convert(Sparse, x), convert(Sparse, s)); k)

# A_mul_B!{A}(k::KUdense{A}, x::KUdense{A}, s::KUsparse{A})=
#     (A_mul_B!(convert(A, k), convert(A, x), convert(Sparse, s)); k)

# At_mul_B!{A}(k::KUdense{A}, x::KUdense{A}, s::KUsparse{A})=
#     (At_mul_B!(convert(A, k), convert(A, x), convert(Sparse, s)); k)

# function A_mul_B!(k::CudaArray{Float32,2}, x::Sparse{CudaArray,Float32,Int32}, s::Sparse{CudaArray,Float32,Int32})
#     @assert size(k)==(size(x,1),size(s,2))
#     ccall((:As_mul_Bs_32,libknet),Void,
#           (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
#           size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
#     return k
# end

# function A_mul_B!(k::CudaArray{Float64,2}, x::Sparse{CudaArray,Float64,Int32}, s::Sparse{CudaArray,Float64,Int32})
#     @assert size(k)==(size(x,1),size(s,2))
#     ccall((:As_mul_Bs_64,libknet),Void,
#           (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
#           size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
#     return k
# end

# This is too slow:
# function At_mul_B!{A<:Array,B<:Array}(k::Matrix, s::Sparse{A}, x::Sparse{B})
#     @assert size(k)==(size(s,2),size(x,2))
#     fillsync!(k, 0)
#     for xcol=1:size(x,2)
#         for scol=1:size(s,2)
#             x1=x.colptr[xcol]; x2=x.colptr[xcol+1]
#             s1=s.colptr[scol]; s2=s.colptr[scol+1]
#             while((x1 < x2) && (s1 < s2))
#                 xrow=x.rowval[x1]; srow=s.rowval[s1]
#                 if (xrow < srow) x1 += 1
#                 elseif (srow < xrow) s1 += 1
#                 else k[scol,xcol] += x.nzval[x1] * s.nzval[s1]; x1+=1; s1+=1
#                 end
#             end
#         end
#     end
#     return k
# end

    # muldbg = nothing
    # C2 = deepcopy(C)
    # A_mul_Btx!(C,A,B)
    # if !isapprox(full(to_host2(C2)), full(to_host2(C)))
    #     global muldbg
    #     muldbg = (A,B,C,C2)
    #     error()
    # end
    # return C2

### KUDENSE

# The input could be a tensor or a vector.  In which case perform
# internal calculations in 2D.

# A_mul_B!{S,T}(C::KUdense{S,T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); gpusync(); C)
# At_mul_B!{S,T}(C::KUdense{S,T}, A::KUdense{S,T}, B::KUdense{S,T})=(At_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); gpusync(); C)
# A_mul_Bt!{S,T}(C::KUdense{S,T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_Bt!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); gpusync(); C)

# # KUdense mixed with other types:
# A_mul_B!{S,T}(C::KUdense{S,T}, A::BaseArray{T}, B::KUdense{S,T})=(A_mul_B!(mat2d(C.arr), mat2d(A), mat2d(B.arr)); gpusync(); C)
# At_mul_B!{S,T}(C::KUdense{S,T}, A::BaseArray{T}, B::KUdense{S,T})=(At_mul_B!(mat2d(C.arr), mat2d(A), mat2d(B.arr)); gpusync(); C)
# A_mul_Bt!{S,T}(C::BaseArray{T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_Bt!(mat2d(C), mat2d(A.arr), mat2d(B.arr)); gpusync(); C)

# axpy!{S,T}(a,x::KUdense{S,T},y::KUdense{S,T})=(axpy!(convert(T,a),x.arr,y.arr); y)
# scale!{S,T}(a,x::KUdense{S,T})=(scale!(convert(T,a),x.arr); x)
