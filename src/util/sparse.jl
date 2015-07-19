using CUDArt
import Base: similar, eltype, length, ndims, size, isempty, copy, copy!
import CUDArt: to_host

# Replicate SparseMatrixCSC but use KUdense for resizeable storage.

type KUsparse{A,T,I<:Integer}
    m::Int                      # Number of rows
    n::Int                      # Number of columns
    colptr::KUdense{A,I,1}      # Column i is in colptr[i]+1:colptr[i+1], note that this is 0 based on cusparse
    rowval::KUdense{A,I,1}      # Row values of nonzeros
    nzval::KUdense{A,T,1}       # Nonzero values
end

KUsparse{A,T,I}(::Type{A}, s::SparseMatrixCSC{T,I})=
    KUsparse{A,T,I}(s.m, s.n, 
                    KUdense(convert(A{I},s.colptr)),
                    KUdense(convert(A{I},s.rowval)),
                    KUdense(convert(A{T},s.nzval)))

KUsparse(S::SparseMatrixCSC)=KUsparse(gpu()?CudaArray:Array, S)
KUsparse{A,T,I}(::Type{A}, ::Type{T}, ::Type{I}, m::Integer, n::Integer)=KUsparse(A,spzeros(T,I,m,n))
KUsparse{A,T,I}(::Type{A}, ::Type{T}, ::Type{I}, d::NTuple{2,Int})=KUsparse(A,T,I,d...)

similar{A,T,I}(s::KUsparse{A}, ::Type{T}, ::Type{I}, m::Integer, n::Integer)=KUsparse(A,T,I,m,n)

### BASIC ARRAY OPS

atype{A}(::KUsparse{A})=A
atype(::SparseMatrixCSC)=Array
itype{A,T,I}(::KUsparse{A,T,I})=I
itype{T,I}(::SparseMatrixCSC{T,I})=I
clength(s::KUsparse)=s.m
clength(s::SparseMatrixCSC)=s.m
eltype{A,T}(::KUsparse{A,T})=T
length(s::KUsparse)=(s.m*s.n)
ndims(::KUsparse)=2
size(s::KUsparse)=(s.m,s.n)
size(s::KUsparse,i)=(i==1?s.m:i==2?s.n:error("Bad dimension"))
isempty(s::KUsparse)=(length(s)==0)
to_host(s::KUsparse{CudaArray})=cpucopy(s)

### BASIC COPY

copy!{A,B,T,I}(a::KUsparse{A,T,I}, b::KUsparse{B,T,I})=
    (a.m=b.m;a.n=b.n;for f in (:colptr,:rowval,:nzval); copy!(a.(f),b.(f)); end; a)

copy!{A,T,I}(a::KUsparse{A,T,I}, b::SparseMatrixCSC{T,I})=
    (a.m=b.m;a.n=b.n;for f in (:colptr,:rowval,:nzval); copy!(a.(f),b.(f)); end; a)

copy(a::KUsparse)=KUsparse(a.m,a.n,copy(a.colptr),copy(a.rowval),copy(a.nzval))


# We need to fix cpu/gpu copy so the type changes appropriately:

cpucopy_internal{T,I}(s::KUsparse{CudaArray,T,I},d::ObjectIdDict)=
    KUsparse{Array,T,I}(s.m,s.n,
                        KUdense(to_host(s.colptr.arr)),
                        KUdense(to_host(s.rowval.arr)),
                        KUdense(to_host(s.nzval.arr)))

gpucopy_internal{T,I}(s::KUsparse{Array,T,I},d::ObjectIdDict)=
    KUsparse{CudaArray,T,I}(s.m,s.n,
                            KUdense(CudaArray(s.colptr.arr)),
                            KUdense(CudaArray(s.rowval.arr)),
                            KUdense(CudaArray(s.nzval.arr)))

