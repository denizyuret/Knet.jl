# TODO: deprecate KUdense in this file.

using Base.LinAlg.BLAS: gemm!, scal!, nrm2
import Base: A_mul_B!, A_mul_Bt!, At_mul_B!, vecnorm
import Base.LinAlg: axpy!, scale!

### VEC functions
axpy!{T}(a,x::CudaArray{T},y::CudaArray{T})=(n=length(x); n==length(y)||error(); axpy!(n,convert(T,a),x,1,y,1); gpusync(); y)
scale!{T}(a,x::CudaArray{T})=(scal!(length(x),convert(T,a),x,1); gpusync(); x)

# CUBLAS is twice as slow as Barret's custom kernel in my experiments:
# vecnorm(x::CudaArray)=nrm2(x)
vecnorm2(x::CudaArray{Float32})=ccall((:vecnorm2_32,libknet),Float32,(Ptr{Cfloat},Cint),x,length(x))
vecnorm2(x::CudaArray{Float64})=ccall((:vecnorm2_64,libknet),Float64,(Ptr{Cdouble},Cint),x,length(x))
vecnorm1(x::CudaArray{Float32})=ccall((:vecnorm1_32,libknet),Float32,(Ptr{Cfloat},Cint),x,length(x))
vecnorm1(x::CudaArray{Float64})=ccall((:vecnorm1_64,libknet),Float64,(Ptr{Cdouble},Cint),x,length(x))
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
mat2d(x)=(ndims(x)==2 ? x : (x2=reshape(x, size2(x));pointer(x2)===pointer(x)||error();x2))
A_mul_B!{T}(C::CudaArray{T},A::CudaArray{T},B::CudaArray{T})=(gemm!('N','N',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); gpusync(); C)
A_mul_Bt!{T}(C::CudaArray{T},A::CudaArray{T},B::CudaArray{T})=(gemm!('N','T',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); gpusync(); C)
At_mul_B!{T}(C::CudaArray{T},A::CudaArray{T},B::CudaArray{T})=(gemm!('T','N',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); gpusync(); C)
A_mul_B!{T}(C::Array{T}, A::Array{T}, B::Array{T})=(gemm!('N','N',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); gpusync(); C)
A_mul_Bt!{T}(C::Array{T}, A::Array{T}, B::Array{T})=(gemm!('N','T',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); gpusync(); C)
At_mul_B!{T}(C::Array{T}, A::Array{T}, B::Array{T})=(gemm!('T','N',one(T),mat2d(A),mat2d(B),zero(T),mat2d(C)); gpusync(); C)

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

# dw = dy * x'
# A_mul_Bt!(csr,dns,csc) is more efficient if we do not insist on
# unique sorted csr entries.  To signal unsorted csr, we introduce a
# new type so we don't accidentally pass it to an unprepared function.
# The following copied from CUSPARSE.jl:

type CudaSparseMatrixCSRU{T}
    rowPtr::CudaArray{Cint,1}
    colVal::CudaArray{Cint,1}
    nzVal::CudaArray{T,1}
    dims::NTuple{2,Int}
    nnz::Cint
    dev::Int
    function CudaSparseMatrixCSRU(rowPtr::CudaVector{Cint}, colVal::CudaVector{Cint}, nzVal::CudaVector{T}, dims::NTuple{2,Int}, nnz::Cint, dev::Int)
        new(rowPtr,colVal,nzVal,dims,nnz,dev)
    end
end

CudaSparseMatrixCSRU(T::Type, m::Integer, n::Integer)=CudaSparseMatrixCSRU{T}(fill!(CudaArray(Cint,m),1), CudaArray(Cint,0), CudaArray(T,0), (convert(Int,m),convert(Int,n)), convert(Cint,0), convert(Int,device()))
CudaSparseMatrixCSRU(T::Type, rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray, dims::NTuple{2,Int}) = CudaSparseMatrixCSRU{T}(rowPtr, colVal, nzVal, dims, convert(Cint,length(nzVal)), device())
CudaSparseMatrixCSRU(T::Type, rowPtr::CudaArray, colVal::CudaArray, nzVal::CudaArray, nnz, dims::NTuple{2,Int}) = CudaSparseMatrixCSRU{T}(rowPtr, colVal, nzVal, dims, nnz, device())
Base.eltype{T}(x::CudaSparseMatrixCSRU{T})=T
Base.size(x::CudaSparseMatrixCSRU)=x.dims
Base.issparse(x::CudaSparseMatrixCSRU)=true
Base.scale!(s,x::CudaSparseMatrixCSRU)=scale!(s,x.nzVal)
Base.similar(Mat::CudaSparseMatrixCSRU) = CudaSparseMatrixCSRU(eltype(Mat), copy(Mat.rowPtr), copy(Mat.colVal), similar(Mat.nzVal), Mat.nnz, Mat.dims)
Base.copy(Mat::CudaSparseMatrixCSRU; stream=null_stream) = copy!(similar(Mat),Mat;stream=null_stream)
function Base.copy!(dst::CudaSparseMatrixCSRU, src::CudaSparseMatrixCSRU; stream=null_stream)
    if dst.dims != src.dims
        throw(ArgumentError("Inconsistent Sparse Matrix size"))
    end
    copy!( dst.rowPtr, src.rowPtr )
    copy!( dst.colVal, src.colVal )
    copy!( dst.nzVal, src.nzVal )
    dst.nnz = src.nnz
    dst
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

function A_mul_Bt_Csize!{T}(C::CudaSparseMatrixCSRU{T},A::CudaMatrix{T},B::CudaSparseMatrixCSC{T})
    C.dims == (size(A,1), size(B,1)) || error()
    C.nnz = C.dims[1] * B.nnz
    resize!(C.rowPtr, C.dims[1]+1)
    resize!(C.colVal, C.nnz)
    resize!(C.nzVal, C.nnz)
end

function A_mul_Bt!(C::CudaSparseMatrixCSRU{Float32},A::CudaMatrix{Float32},B::CudaSparseMatrixCSC{Float32})
    A_mul_Bt_Csize!(C,A,B)
    ccall((:mul_dns_csr_csr_32,libknet), Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          size(A,1), size(A,2), A, B.colPtr, B.rowVal, B.nzVal, C.rowPtr, C.colVal, C.nzVal)
    gpusync()
    return C
end

function A_mul_Bt!(C::CudaSparseMatrixCSRU{Float64},A::CudaMatrix{Float64},B::CudaSparseMatrixCSC{Float64})
    A_mul_Bt_Csize!(C,A,B)
    ccall((:mul_dns_csr_csr_64,libknet), Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          size(A,1), size(A,2), A, B.colPtr, B.rowVal, B.nzVal, C.rowPtr, C.colVal, C.nzVal)
    gpusync()
    return C
end

# dw = dy * x'
# This version gives a proper CSR matrix but is slower:
function A_mul_Bt!{T}(C::CudaSparseMatrixCSR{T},A::CudaMatrix{T},B::CudaSparseMatrixCSC{T})
    bT = CudaSparseMatrixCSR{T}(B.colPtr, B.rowVal, B.nzVal, (B.dims[2],B.dims[1]), B.nnz, B.dev)
    a = sparse(A)               # t:337 gives CudaSparseMatrixCSR
    gemm!('N','N',a,bT,C)       # t:868
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
Base.fill!(x::ArrayAccumulator, n)=(n==0 ? x.cnt = 0 : error())
Base.scale!(s, x::ArrayAccumulator)=(for i=1:x.cnt; scale!(s,x.arr[i]); end; x)

# TODO: this does not work for unsorted csru or ArrayAccumulator,
# since they may have multiple entries for a single position.  Note
# that p=1 is not feasible either since entries might cancel out.
# Base.vecnorm(x::ArrayAccumulator)=(n=0;for i=1:x.cnt;
# n+=vecnorm(x.arr[i]); end; n)
function Base.vecnorm(x::Union{ArrayAccumulator,CudaSparseMatrixCSRU},p=2)
    Base.warn_once("Cannot compute vecnorm for $(typeof(x)), returning 0")
    return 0
end

function axpy!(a, x, y::ArrayAccumulator)
    @assert size(x)==size(y) && eltype(x)==eltype(y)
    y.cnt += 1
    if length(y.arr) >= y.cnt
        copy!(y.arr[y.cnt], x)
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


axpy!(a,x::CudaSparseMatrixCSRU{Float32},y::CudaMatrix{Float32})=(ccall((:add_csr_dns_32,libknet),Void,(Cint,Cint,Cfloat,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),x.dims[1],x.dims[2],convert(Float32,a),x.nnz,x.nzVal,x.rowPtr,x.colVal,y); gpusync(); y)
axpy!(a,x::CudaSparseMatrixCSRU{Float64},y::CudaMatrix{Float64})=(ccall((:add_csr_dns_64,libknet),Void,(Cint,Cint,Cdouble,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),x.dims[1],x.dims[2],convert(Float64,a),x.nnz,x.nzVal,x.rowPtr,x.colVal,y); gpusync(); y)


### axpb! useful scale and shift transformation: x -> ax+b

axpb!(a::Number, b::Number, x::Array)=(for i=1:length(x); x[i]=a*x[i]+b; end; x)
axpb!(a::Number, b::Number, x::CudaArray{Float32})=(ccall((:axpb32,libknet),Void,(Cint,Cfloat,Cfloat,Ptr{Cfloat}),length(x),a,b,x); gpusync(); x)
axpb!(a::Number, b::Number, x::CudaArray{Float64})=(ccall((:axpb64,libknet),Void,(Cint,Cdouble,Cdouble,Ptr{Cdouble}),length(x),a,b,x); gpusync(); x)


### mul2 element-wise multiplication:

# mul2!(c::KUdense,a::KUdense,b::KUdense)=(mul2!(c.arr,a.arr,b.arr);c)
mul2!(c::Array,a::Array,b::Array)=(for i=1:length(c); c[i] = a[i]*b[i]; end; c)
mul2!(c::CudaArray{Float32},a::CudaArray{Float32},b::CudaArray{Float32})=(ccall((:mul2_32,libknet),Void,(Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),length(a),a,b,c); gpusync(); c)
mul2!(c::CudaArray{Float64},a::CudaArray{Float64},b::CudaArray{Float64})=(ccall((:mul2_64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),length(a),a,b,c); gpusync(); c)



### DEAD CODE:


### SPARSE: A_mul_Bs!(y, w, x)

# A_mul_B!{A<:Array,B<:Array}(y::KUdense{A}, w::Array, x::KUsparse{B})=(A_mul_B!(y.arr, w, x); y)
# A_mul_B!{A<:CudaArray,B<:CudaArray}(y::KUdense{A}, w::CudaArray, x::KUsparse{B})=(A_mul_B!(y.arr, w, x); y)

# function A_mul_B!{A<:Array}(y::Matrix, w::Matrix, x::KUsparse{A}) # 1607
#     @assert size(y)==(size(w,1), size(x,2))
#     # eltype's do not have to match.
#     fill!(y, zero(eltype(y)))
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
#     fill!(dw, zero(eltype(dw)))
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

# At_mul_B!(k::Matrix, s::SparseMatrixCSC, x::SparseMatrixCSC)=copy!(k, s' * x)

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
#     fill!(k, 0)
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
