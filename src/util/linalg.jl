# TODO: deprecate KUdense in this file.

using Base.LinAlg.BLAS: gemm!, scal!, nrm2
import Base: A_mul_B!, A_mul_Bt!, At_mul_B!, vecnorm
import Base.LinAlg: axpy!, scale!

### VEC functions
# axpy!{S,T}(a,x::KUdense{S,T},y::KUdense{S,T})=(axpy!(convert(T,a),x.arr,y.arr); y)
axpy!{T}(a,x::CudaArray{T},y::CudaArray{T})=(n=length(x); @assert n==length(y); axpy!(n,convert(T,a),x,1,y,1); y)
# scale!{S,T}(a,x::KUdense{S,T})=(scale!(convert(T,a),x.arr); x)
scale!{T}(a,x::CudaArray{T})=(scal!(length(x),convert(T,a),x,1); x)
vecnorm(x::CudaArray)=nrm2(x)

### MMUL
# This is not a complete implementation.  The goal is to support Knet
# operations for sparse/dense matrices on cpu/gpu.  The operations needed:
#
# mmul forw: A_mul_B!(y, w, x)		A_mul_Bs!(y, w, x): cpu/gpu: kudense, array, sparse
# mmul back: A_mul_Bt!(dw, dy, x)	A_mul_Bst!(dw, dy, x): cpu/gpu: array, kudense, sparse
# mmul back: At_mul_B!(dx, w, dy)	no dx: only initial input can be sparse
# kper forw: At_mul_B!(k, s, x)		Ast_mul_Bs!(k, s, x): cpu/gpu: kudense, kusparse, sparse


### CUDAARRAY (Array versions already defined)

A_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','N',one(T),A,B,zero(T),C)
A_mul_Bt!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','T',one(T),A,B,zero(T),C)
At_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('T','N',one(T),A,B,zero(T),C)

### KUDENSE

# The input could be a tensor or a vector.  In which case perform
# internal calculations in 2D.

mat2d(x)=(ndims(x)==2 ? x : reshape(x, size2(x)))
# A_mul_B!{S,T}(C::KUdense{S,T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
# At_mul_B!{S,T}(C::KUdense{S,T}, A::KUdense{S,T}, B::KUdense{S,T})=(At_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
# A_mul_Bt!{S,T}(C::KUdense{S,T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_Bt!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)

# # KUdense mixed with other types:
# A_mul_B!{S,T}(C::KUdense{S,T}, A::BaseArray{T}, B::KUdense{S,T})=(A_mul_B!(mat2d(C.arr), mat2d(A), mat2d(B.arr)); C)
# At_mul_B!{S,T}(C::KUdense{S,T}, A::BaseArray{T}, B::KUdense{S,T})=(At_mul_B!(mat2d(C.arr), mat2d(A), mat2d(B.arr)); C)
# A_mul_Bt!{S,T}(C::BaseArray{T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_Bt!(mat2d(C), mat2d(A.arr), mat2d(B.arr)); C)

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
    return C
end

# dw = dy * x'
function A_mul_Bt!{T}(C::CudaSparseMatrixCSR{T},A::CudaMatrix{T},B::CudaSparseMatrixCSC{T})
    bT = CudaSparseMatrixCSR{T}(B.colPtr, B.rowVal, B.nzVal, (B.dims[2],B.dims[1]), B.nnz, B.dev)
    a = sparse(A)           # gives CudaSparseMatrixCSR
    gemm!('N','N',a,bT,C)
    free(a)
    return C
end

# TODO: the sparsity pattern should be identical here, could make this much faster:
# iw = dy * x'  : we have the same x, and dy is always dense
# if that is the case we just do axpy!(1,x.nzVal,y.nzVal)
function axpy!{T}(a,x::CudaSparseMatrixCSR{T},y::CudaSparseMatrixCSR{T})
    z = CUSPARSE.geam(convert(T,a), x, one(T), y, 'O', 'O', 'O')
    free(y)
    y.rowPtr = z.rowPtr
    y.colVal = z.colVal
    y.nzVal = z.nzVal
    y.nnz = z.nnz
    return y
end

axpy!(a,x::CudaSparseMatrixCSR{Float32},y::CudaMatrix{Float32})=(ccall((:axpy32csr,libknet),Void,(Cint,Cint,Cfloat,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),x.dims[1],x.dims[2],convert(Float32,a),x.nnz,x.nzVal,x.rowPtr,x.colVal,y); y)
axpy!(a,x::CudaSparseMatrixCSR{Float64},y::CudaMatrix{Float64})=(ccall((:axpy64csr,libknet),Void,(Cint,Cint,Cdouble,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),x.dims[1],x.dims[2],convert(Float64,a),x.nnz,x.nzVal,x.rowPtr,x.colVal,y); y)


### axpb! useful scale and shift transformation: x -> ax+b

axpb!(a::Number, b::Number, x::Array)=(for i=1:length(x); x[i]=a*x[i]+b; end; x)
axpb!(a::Number, b::Number, x::CudaArray{Float32})=(ccall((:axpb32,libknet),Void,(Cint,Cfloat,Cfloat,Ptr{Cfloat}),length(x),a,b,x);x)
axpb!(a::Number, b::Number, x::CudaArray{Float64})=(ccall((:axpb64,libknet),Void,(Cint,Cdouble,Cdouble,Ptr{Cdouble}),length(x),a,b,x);x)


### mul2 element-wise multiplication:

# mul2!(c::KUdense,a::KUdense,b::KUdense)=(mul2!(c.arr,a.arr,b.arr);c)
mul2!(c::Array,a::Array,b::Array)=(for i=1:length(c); c[i] = a[i]*b[i]; end; c)
mul2!(c::CudaArray{Float32},a::CudaArray{Float32},b::CudaArray{Float32})=(ccall((:mul2_32,libknet),Void,(Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),length(a),a,b,c);c)
mul2!(c::CudaArray{Float64},a::CudaArray{Float64},b::CudaArray{Float64})=(ccall((:mul2_64,libknet),Void,(Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),length(a),a,b,c);c)



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
