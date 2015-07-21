using Base.LinAlg.BLAS: gemm!, scal!
import Base: A_mul_B!, A_mul_Bt!, At_mul_B!
import Base.LinAlg: axpy!, scale!

### AXPY! and SCALE!
axpy!{S,T}(a,x::KUdense{S,T},y::KUdense{S,T})=(axpy!(convert(T,a),x.arr,y.arr); y)
axpy!{T}(a,x::CudaArray{T},y::CudaArray{T})=(n=length(x); @assert n==length(y); axpy!(n,convert(T,a),x,1,y,1); y)
scale!{S,T}(a,x::KUdense{S,T})=(scale!(convert(T,a),x.arr); x)
scale!{T}(a,x::CudaArray{T})=(scal!(n,convert(T,a),x,1); x)

### MMUL
# This is not a complete implementation.  The goal is to support KUnet
# operations for sparse/dense matrices on cpu/gpu.  The operations needed:
#
# mmul forw: A_mul_B!(y, w, x)		A_mul_Bs!(y, w, x): cpu/gpu: kudense, array, kusparse
# mmul back: A_mul_Bt!(dw, dy, x)	A_mul_Bst!(dw, dy, x): cpu/gpu: array, kudense, kusparse
# mmul back: At_mul_B!(dx, w, dy)	no dx: only initial input can be sparse
# kper forw: At_mul_B!(k, s, x)		Ast_mul_Bs!(k, s, x): cpu/gpu: kudense, kusparse, kusparse


### CUDAARRAY (Array versions already defined)

A_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','N',one(T),A,B,zero(T),C)
A_mul_Bt!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('N','T',one(T),A,B,zero(T),C)
At_mul_B!{T}(C::CudaArray{T,2}, A::CudaArray{T,2}, B::CudaArray{T,2})=gemm!('T','N',one(T),A,B,zero(T),C)

### KUDENSE

# The input could be a tensor or a vector.  In which case perform
# internal calculations in 2D.

mat2d(x)=(ndims(x)==2 ? x : reshape(x, size2(x)))
A_mul_B!{S,T}(C::KUdense{S,T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
At_mul_B!{S,T}(C::KUdense{S,T}, A::KUdense{S,T}, B::KUdense{S,T})=(At_mul_B!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)
A_mul_Bt!{S,T}(C::KUdense{S,T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_Bt!(mat2d(C.arr), mat2d(A.arr), mat2d(B.arr)); C)

# KUdense mixed with other types:
A_mul_B!{S,T}(C::KUdense{S,T}, A::BaseArray{T}, B::KUdense{S,T})=(A_mul_B!(mat2d(C.arr), mat2d(A), mat2d(B.arr)); C)
At_mul_B!{S,T}(C::KUdense{S,T}, A::BaseArray{T}, B::KUdense{S,T})=(At_mul_B!(mat2d(C.arr), mat2d(A), mat2d(B.arr)); C)
A_mul_Bt!{S,T}(C::BaseArray{T}, A::KUdense{S,T}, B::KUdense{S,T})=(A_mul_Bt!(mat2d(C), mat2d(A.arr), mat2d(B.arr)); C)

### SPARSE: A_mul_Bs!(y, w, x)

function A_mul_B!(y::KUdense{Array}, w::Array, x::KUsparse{Array})
    @assert size(y)==(size(w,1), size(x,2))
    A_mul_B!(convert(Array, y), w, convert(Sparse, x))
    return y
end

function A_mul_B!(y::KUdense{CudaArray}, w::CudaArray, x::KUsparse{CudaArray})
    @assert size(y)==(size(w,1), size(x,2))
    A_mul_B!(convert(CudaArray, y), w, convert(Sparse, x))
    return y
end

function A_mul_B!(y::Matrix, w::Matrix, x::Sparse{Array}) # 1607
    @assert size(y)==(size(w,1), size(x,2))
    # eltype's do not have to match.
    fill!(y, zero(eltype(y)))
    @inbounds for xcol=1:size(x,2)
        @inbounds for xp=x.colptr[xcol]:(x.colptr[xcol+1]-1)
            xval = x.nzval[xp]  # 133
            xrow = x.rowval[xp] # wcol
            @inbounds for wrow=1:size(w,1)
                wval = w[wrow,xrow]
                yinc = wval * xval  # 245
                y[wrow,xcol] += yinc # 789
            end
        end
    end
    return y
end

function A_mul_B!(y::CudaArray{Float32,2}, w::CudaArray{Float32,2}, x::Sparse{CudaArray,Float32,Int32})
    @assert size(y)==(size(w,1),size(x,2))
    ccall((:A_mul_Bs_32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          size(w,1),size(x,2),w,x.nzval,x.rowval,x.colptr,y)
    return y
end

function A_mul_B!(y::CudaArray{Float64,2}, w::CudaArray{Float64,2}, x::Sparse{CudaArray,Float64,Int32})
    @assert size(y)==(size(w,1),size(x,2))
    ccall((:A_mul_Bs_64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          size(w,1),size(x,2),w,x.nzval,x.rowval,x.colptr,y)
    return y
end

### SPARSE: A_mul_Bst!(dw, dy, x)

function A_mul_Bt!(dw::Array, dy::KUdense{Array}, x::KUsparse{Array})
    @assert size(dw)==(size(dy,1), size(x,1))
    A_mul_Bt!(dw, convert(Array, dy), convert(Sparse,x))
    return dw
end

function A_mul_Bt!(dw::CudaArray, dy::KUdense{CudaArray}, x::KUsparse{CudaArray})
    @assert size(dw)==(size(dy,1), size(x,1))
    A_mul_Bt!(dw, convert(CudaArray, dy), convert(Sparse,x))
    return dw
end

function A_mul_Bt!(dw::Matrix, dy::Matrix, x::Sparse{Array})
    @assert size(dw)==(size(dy,1), size(x,1))
    fill!(dw, zero(eltype(dw)))
    @inbounds for xcol=1:size(x,2)                      # xcol = ycol
        xrange = x.colptr[xcol]:(x.colptr[xcol+1]-1)	
        @inbounds for xp in xrange
            xrow = x.rowval[xp]                         # xrow = wcol
            xval = x.nzval[xp]
            @inbounds for yrow=1:size(dy,1)             # yrow = wrow
                yval = dy[yrow,xcol]
                winc = xval * yval
                dw[yrow,xrow] += winc
            end
        end
    end
    return dw
end

function A_mul_Bt!(dw::CudaArray{Float32,2}, dy::CudaArray{Float32,2}, x::Sparse{CudaArray,Float32,Int32})
    @assert size(dw)==(size(dy,1),size(x,1))
    ccall((:A_mul_Bst_32,libkunet),Void,
          (Cint,Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          size(dy,1),size(dy,2),size(x,1),dy,x.nzval,x.rowval,x.colptr,dw)
    return dw
end

function A_mul_Bt!(dw::CudaArray{Float64,2}, dy::CudaArray{Float64,2}, x::Sparse{CudaArray,Float64,Int32})
    @assert size(dw)==(size(dy,1),size(x,1))
    ccall((:A_mul_Bst_64,libkunet),Void,
          (Cint,Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          size(dy,1),size(dy,2),size(x,1),dy,x.nzval,x.rowval,x.colptr,dw)
    return dw
end


### SPARSE: Ast_mul_Bs!(k, s, x)

function At_mul_B!{A}(k::KUdense{A}, s::KUsparse{A}, x::KUsparse{A})
    @assert size(k)==(size(s,2),size(x,2))
    At_mul_B!(convert(A, k), convert(Sparse, s), convert(Sparse, x))
    return k
end

function At_mul_B!(k::Matrix, s::Sparse{Array}, x::Sparse{Array})
    @assert size(k)==(size(s,2),size(x,2))
    fill!(k, zero(eltype(k)))
    @inbounds for xcol=1:size(x,2)
        @inbounds for scol=1:size(s,2)
            x1=x.colptr[xcol]; x2=x.colptr[xcol+1]
            s1=s.colptr[scol]; s2=s.colptr[scol+1]
            while((x1 < x2) && (s1 < s2))
                xrow=x.rowval[x1]; srow=s.rowval[s1]
                if (xrow < srow) x1 += 1
                elseif (srow < xrow) s1 += 1
                else k[scol,xcol] += x.nzval[x1] * s.nzval[s1]; x1+=1; s1+=1
                end
            end
        end
    end
    return k
end

function At_mul_B!(k::CudaArray{Float32,2}, s::Sparse{CudaArray,Float32,Int32}, x::Sparse{CudaArray,Float32,Int32})
    @assert size(k)==(size(s,2),size(x,2))
    ccall((:Ast_mul_Bs_32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          size(s,2),size(x,2),s.nzval,s.rowval,s.colptr,x.nzval,x.rowval,x.colptr,k)
    return k
end

function At_mul_B!(k::CudaArray{Float64,2}, s::Sparse{CudaArray,Float64,Int32}, x::Sparse{CudaArray,Float64,Int32})
    @assert size(k)==(size(s,2),size(x,2))
    ccall((:Ast_mul_Bs_64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          size(s,2),size(x,2),s.nzval,s.rowval,s.colptr,x.nzval,x.rowval,x.colptr,k)
    return k
end

### DEAD CODE:

### KUSPARSE

# A_mul_B!{A}(k::KUdense{A}, x::KUsparse{A}, s::KUsparse{A})=
#     (A_mul_B!(convert(A, k), convert(Sparse, x), convert(Sparse, s)); k)

# A_mul_B!{A}(k::KUdense{A}, x::KUdense{A}, s::KUsparse{A})=
#     (A_mul_B!(convert(A, k), convert(A, x), convert(Sparse, s)); k)

# At_mul_B!{A}(k::KUdense{A}, x::KUdense{A}, s::KUsparse{A})=
#     (At_mul_B!(convert(A, k), convert(A, x), convert(Sparse, s)); k)

# function A_mul_B!(k::CudaArray{Float32,2}, x::Sparse{CudaArray,Float32,Int32}, s::Sparse{CudaArray,Float32,Int32})
#     @assert size(k)==(size(x,1),size(s,2))
#     ccall((:As_mul_Bs_32,libkunet),Void,
#           (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
#           size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
#     return k
# end

# function A_mul_B!(k::CudaArray{Float64,2}, x::Sparse{CudaArray,Float64,Int32}, s::Sparse{CudaArray,Float64,Int32})
#     @assert size(k)==(size(x,1),size(s,2))
#     ccall((:As_mul_Bs_64,libkunet),Void,
#           (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
#           size(x,1),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k)
#     return k
# end

