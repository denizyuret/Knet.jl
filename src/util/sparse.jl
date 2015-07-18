# Replicate SparseMatrixCSC but use KUdense for resizeable storage.

type KUsparse{A,T}
    m::Int                      # Number of rows
    n::Int                      # Number of columns
    colptr::KUdense{A,Int32,1}  # Column i is in colptr[i]+1:colptr[i+1], note that this is 0 based on cusparse
    rowval::KUdense{A,Int32,1}  # Row values of nonzeros
    nzval::KUdense{A,T,1}       # Nonzero values
end

KUsparse{T}(A::Type, S::SparseMatrixCSC{T})=
    KUsparse{A,T}(s.m, s.n, 
                  KUdense(convert(A{Int32},s.colptr)),
                  KUdense(convert(A{Int32},s.rowval)),
                  KUdense(convert(A{T},s.nzval)))

KUsparse(S::SparseMatrixCSC)=KUsparse(gpu()?CudaArray:Array, S)
KUsparse(A::Type, T::Type, m::Integer, n::Integer)=KUsparse(A,spzeros(T,Int32,m,n))
KUsparse(A::Type, T::Type, d::NTuple{2,Int})=KUsparse(A,T,d...)
Base.similar{A}(s::KUsparse{A}, T, m, n)=KUsparse(A,T,m,n)

### BASIC ARRAY OPS

atype{A}(::KUsparse{A})=A
Base.eltype{A,T}(::KUsparse{A,T})=T
Base.length(s::KUsparse)=(s.m*s.n)
Base.ndims(::KUsparse)=2
Base.size(s::KUsparse)=(s.m,s.n)
Base.size(s::KUsparse,i)=(i==1?s.m:i==2?s.n:error("Bad dimension"))
Base.isempty(s::KUsparse)=(length(s)==0)
clength(s::KUsparse)=s.m

# We need to fix cpu/gpu copy so the type changes appropriately:
cpucopy_internal{T}(s::KUsparse{CudaArray,T},d::ObjectIdDict)=
    KUsparse{Array,T}(s.m,s.n,to_host(s.colptr),to_host(s.rowval),to_host(nzval))

gpucopy_internal{T}(s::KUsparse{Array,T},d::ObjectIdDict)=
    KUsparse{CudaArray,T}(s.m,s.n,CudaArray(s.colptr),CudaArray(s.rowval),CudaArray(nzval))


