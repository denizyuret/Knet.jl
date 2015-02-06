# TODO
# make it work without cudart and gpu
# debug extra memory usage in gpu mode: is that why blas.jl works with pointers?  ccall convert problem?
# install update using axpy!


### Union types to cover both regular and cuda arrays:
using CUDArt: CudaArray, CudaMatrix, CudaVector
typealias Mat{t} Union(AbstractMatrix{t}, CudaMatrix{t})
typealias Vec{t} Union(AbstractVector{t}, CudaVector{t})
typealias Arr{t} Union(AbstractArray{t}, CudaArray{t})

### Defaults for regular arrays:
import CUDArt: to_host, free
to_host(x)=x
free(x)=x

### To help debug memory
mysizeof(x)=sizeof(eltype(x))*length(x)

### We need some utilities for CudaArrays:
import Base: copy!, isempty
copy!{T}(a::CudaArray{T}, b::SubArray{T}) = copy!(a, map(d->1:d,size(a)), b.parent, b.indexes)
isempty(a::CudaArray)=(length(a)==0)

### CUDA versions of activation functions:

const libjnet="./libjnet.so"

function reluforw(y::CudaMatrix{Float32})
    ccall(("reluforw",libjnet), Void, (Ptr{Float32},Cint), y, length(y))
end

function reluback(dy::CudaMatrix{Float32}, y::CudaMatrix{Float32})
    ccall(("reluback",libjnet), Void, (Ptr{Float32},Ptr{Float32},Cint), dy, y, length(y))
end

function softback(dy::Mat{Float32}, y::CudaMatrix{Float32})
    ddy = isa(dy, CudaArray) ? dy : CudaArray(dy)
    ccall(("softback",libjnet), Void, (Ptr{Float32},Ptr{Float32}, Cint,Cint), ddy, y, size(y,1), size(y,2))
    if (!is(ddy,dy)) free(ddy); end
end

import Base.fill!

function fill!(y::CudaArray{Float32}, val::Float32)
    ccall(("fill",libjnet), Void, (Ptr{Float32},Cint,Float32), y, length(y), val)
end


### We need blas operations for CudaArrays:
import Base.LinAlg.BLAS: gemm!, ger!, gemv!, axpy!
blas_set_num_threads(12)
using Base.LinAlg: BlasChar, BlasInt
libcublas="libcublas"

## gemm: C = alpha A*B + beta C
for (fname, elty) in
        (("cublasDgemm",:Float64),
         ("cublasSgemm",:Float32),
         ("cublasZgemm",:Complex128),
         ("cublasCgemm",:Complex64))
    @eval begin
        function gemm!(transA::BlasChar, transB::BlasChar, alpha::($elty), 
                       A::Mat{$elty}, B::Mat{$elty}, beta::($elty), C::Mat{$elty})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch())
            end
            dA = isa(A, CudaArray) ? A : CudaArray(A)
            dB = isa(B, CudaArray) ? B : CudaArray(B)
            dC = isa(C, CudaArray) ? C : CudaArray(C)
            ccall(($(fname), $(libcublas)), Void,
                  (BlasChar, BlasChar, BlasInt, BlasInt, BlasInt, $elty, Ptr{$elty}, BlasInt, 
                  Ptr{$elty}, BlasInt, $elty, Ptr{$elty}, BlasInt),
                  transA, transB, m, n, k, alpha, pointer(dA), max(1,stride(A,2)), 
                  pointer(dB), max(1,stride(B,2)), beta, pointer(dC), max(1,stride(C,2)))
            if (!is(dA,A)) free(dA); end
            if (!is(dB,B)) free(dB); end
            if (!is(dC,C)) copy!(C,dC); free(dC); end
            C
        end
    end
end

### ger: A = α x y' + A
for (fname, elty) in 
    (("cublasDger",:Float64),
     ("cublasSger",:Float32),
     ("cublasZger",:Complex128),
     ("cublasCger",:Complex64))
    @eval begin
        function ger!(α::$elty, x::Vec{$elty}, y::Vec{$elty}, A::Mat{$elty})
            m, n = size(A)
            m == length(x) || throw(DimensionMismatch())
            n == length(y) || throw(DimensionMismatch())
            dA = isa(A, CudaArray) ? A : CudaArray(A)
            dx = isa(x, CudaArray) ? x : CudaArray(x)
            dy = isa(y, CudaArray) ? y : CudaArray(y)
            ccall(($(fname), $(libcublas)), Void,
                (BlasInt, BlasInt, $elty, Ptr{$elty},
                 BlasInt, Ptr{$elty}, BlasInt, Ptr{$elty}, BlasInt),
                 m, n, α, pointer(dx), 1, 
                 pointer(dy), 1, pointer(dA), max(1,stride(dA,2)))
            if (!is(dA,A)) copy!(A,dA); free(dA); end
            if (!is(dx,x)) free(dx); end
            if (!is(dy,y)) free(dy); end
            A
        end
    end
end


### gemv: y = alpha trans(A) x + beta y
for (fname, elty) in
        (("cublasDgemv",:Float64),
         ("cublasSgemv",:Float32),
         ("cublasZgemv",:Complex128),
         ("cublasCgemv",:Complex64))
    @eval begin
        function gemv!(trans::BlasChar, alpha::($elty), A::Mat{$elty}, X::Vec{$elty}, beta::($elty), Y::Vec{$elty})
            m,n = size(A)
            length(X) == (trans == 'N' ? n : m) && length(Y) == (trans == 'N' ? m : n) || throw(DimensionMismatch())
            dA = isa(A, CudaArray) ? A : CudaArray(A)
            dX = isa(X, CudaArray) ? X : CudaArray(X)
            dY = isa(Y, CudaArray) ? Y : CudaArray(Y)
            ccall(($(fname), $(libcublas)), Void,
                (BlasChar, BlasInt, BlasInt, $elty,
                 Ptr{$elty}, BlasInt, Ptr{$elty}, BlasInt,
                 $elty, Ptr{$elty}, BlasInt),
                 trans, size(dA,1), size(dA,2), alpha,
                 dA, max(1,stride(dA,2)), dX, stride(dX,1),
                 beta, dY, stride(dY,1))
            if (!is(dA,A)) free(dA); end
            if (!is(dX,X)) free(dX); end
            if (!is(dY,Y)) copy!(Y,dY); free(dY); end
            Y
        end
    end
end


### axpy: y = a*x+y
for (fname, elty) in
    (("cublasDaxpy",:Float64),
     ("cublasSaxpy",:Float32),
     ("cublasZaxpy",:Complex128),
     ("cublasCaxpy",:Complex64))
    @eval begin
    function axpy!(alpha::($elty), X::Arr{$elty}, Y::Arr{$elty})
	length(X) == length(Y) || throw(DimensionMismatch("x has length $length(X), but y has length $length(Y)"))
        dX = isa(X, CudaArray) ? X : CudaArray(X)
        dY = isa(Y, CudaArray) ? Y : CudaArray(Y)
        ccall(($(fname),$(libcublas)), Void,
              (BlasInt, $elty, Ptr{$elty}, BlasInt, Ptr{$elty}, BlasInt),
	      length(X), alpha, dX, stride(X,1), dY, stride(Y,1))
        if (!is(dX,X)) free(dX); end
        if (!is(dY,Y)) copy!(Y,dY); free(dY); end
        Y
    end
  end
end
