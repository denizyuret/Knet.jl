using CUDArt
import Base: convert, similar, copy, copy!, eltype, length, ndims, size, isempty, issparse, stride, strides
import CUDArt: to_host

# I want to make the base array explicit in the type signature of sparse arrays:
# So instead of SparseMatrixCSC{T,I} we use the equivalent Sparse{Array,T,I}:

type Sparse{A,T,I<:Integer}; m; n; colptr; rowval; nzval; end

convert{T,I}(::Type{Sparse}, s::SparseMatrixCSC{T,I})=convert(Sparse{Array}, s)

convert{A,T,I}(::Type{Sparse{A}}, s::SparseMatrixCSC{T,I})=
    Sparse{A,T,I}(s.m,s.n,
                  convert(A, s.colptr),
                  convert(A, s.rowval),
                  convert(A, s.nzval))

convert{A,T,I}(::Type{SparseMatrixCSC}, s::Sparse{A,T,I})=
    SparseMatrixCSC(s.m,s.n,
                    convert(Array, s.colptr),
                    convert(Array, s.rowval),
                    convert(Array, s.nzval))

convert{T}(::Type{Sparse}, a::Array{T,2})=convert(Sparse{Array}, a)
convert{T}(::Type{Sparse}, a::CudaArray{T,2})=convert(Sparse{CudaArray}, a)

convert{T}(::Type{Sparse{Array}}, a::BaseArray{T,2})=
    convert(Sparse, convert(SparseMatrixCSC{T,Int32}, sparse(convert(Array, a))))

convert{T}(::Type{Sparse{CudaArray}}, a::BaseArray{T,2})=
    gpucopy(convert(Sparse{Array}, a))

# Now we can construct a Sparse{CudaArray,T,I} using gpucopy:

cpucopy_internal{T,I}(s::Sparse{CudaArray,T,I},d::ObjectIdDict)=
    (haskey(d,s) ? d[s] : 
     Sparse{Array,T,I}(s.m, s.n,
                       cpucopy_internal(s.colptr,d),
                       cpucopy_internal(s.rowval,d),
                       cpucopy_internal(s.nzval,d)))

gpucopy_internal{T,I}(s::Sparse{Array,T,I},d::ObjectIdDict)=
    (haskey(d,s) ? d[s] : 
     Sparse{CudaArray,T,I}(s.m,s.n,
                           gpucopy_internal(s.colptr,d),
                           gpucopy_internal(s.rowval,d),
                           gpucopy_internal(s.nzval,d)))

# And we can construct KUsparse which uses resizeable KUdense arrays for members:

type KUsparse{A,T,I<:Integer}
    m::Int                      # Number of rows
    n::Int                      # Number of columns
    colptr::KUdense{A,I,1}      # Column i is in colptr[i]+1:colptr[i+1], note that this is 0 based on cusparse
    rowval::KUdense{A,I,1}      # Row values of nonzeros
    nzval::KUdense{A,T,1}       # Nonzero values
end

KUsparse{A,T,I}(::Type{A}, ::Type{T}, ::Type{I}, m::Integer, n::Integer)=convert(KUsparse{A}, spzeros(T,I,m,n))

KUsparse{A,T,I}(::Type{A}, ::Type{T}, ::Type{I}, d::NTuple{2,Int})=KUsparse(A,T,I,d...)

similar{A,T,I}(s::KUsparse{A}, ::Type{T}, ::Type{I}, m::Integer, n::Integer)=KUsparse(A,T,I,m,n)

similar{A,T,I,U}(s::KUsparse{A,T,I}, ::Type{U}, d::NTuple{2,Int})=KUsparse(A,U,I,d...)

convert{A,T,I}(::Type{KUsparse}, s::Sparse{A,T,I})=
    KUsparse{A,T,I}(s.m,s.n,convert(KUdense,s.colptr),convert(KUdense,s.rowval),convert(KUdense,s.nzval))

convert{A,T,I}(::Type{Sparse}, s::KUsparse{A,T,I})=
    Sparse{A,T,I}(s.m,s.n,convert(A,s.colptr),convert(A,s.rowval),convert(A,s.nzval))

convert{A,T,I}(::Type{SparseMatrixCSC}, s::KUsparse{A,T,I})=
    SparseMatrixCSC(s.m,s.n,convert(Array,s.colptr),convert(Array,s.rowval),convert(Array,s.nzval))

convert{A,T,I}(::Type{KUsparse{A}}, s::SparseMatrixCSC{T,I})=
    KUsparse{A,T,I}(s.m,s.n,
                    convert(KUdense{A},s.colptr),
                    convert(KUdense{A},s.rowval),
                    convert(KUdense{A},s.nzval))

convert{T,I}(::Type{KUsparse}, s::SparseMatrixCSC{T,I})=convert(KUsparse{Array}, s)

convert(::Type{SparseMatrixCSC}, a::Array)=sparse(a)

convert{T}(::Type{KUsparse}, a::Array{T,2})=convert(KUsparse{Array}, a)
convert{T}(::Type{KUsparse}, a::CudaArray{T,2})=convert(KUsparse{CudaArray}, a)

convert{T}(::Type{KUsparse{Array}}, a::BaseArray{T,2})=
    convert(KUsparse, convert(SparseMatrixCSC{T,Int32}, sparse(convert(Array, a))))

convert{T}(::Type{KUsparse{CudaArray}}, a::BaseArray{T,2})=
    gpucopy(convert(KUsparse{Array}, a))

### BASIC COPY

copy!{A,B,T,I}(a::KUsparse{A,T,I}, b::KUsparse{B,T,I})=
    (a.m=b.m;a.n=b.n;for f in (:colptr,:rowval,:nzval); copy!(a.(f),b.(f)); end; a)

copy!{A,T,I}(a::KUsparse{A,T,I}, b::SparseMatrixCSC{T,I})=
    (a.m=b.m;a.n=b.n;for f in (:colptr,:rowval,:nzval); copy!(a.(f),b.(f)); end; a)

copy(a::KUsparse)=KUsparse(a.m,a.n,copy(a.colptr),copy(a.rowval),copy(a.nzval))


# We need to fix cpu/gpu copy so the type changes appropriately:

cpucopy_internal{T,I}(s::KUsparse{CudaArray,T,I},d::ObjectIdDict)=
    (haskey(d,s) ? d[s] :
     KUsparse{Array,T,I}(s.m,s.n,
                         cpucopy_internal(s.colptr, d),
                         cpucopy_internal(s.rowval, d),
                         cpucopy_internal(s.nzval, d)))

gpucopy_internal{T,I}(s::KUsparse{Array,T,I},d::ObjectIdDict)=
    (haskey(d,s) ? d[s] : 
     KUsparse{CudaArray,T,I}(s.m,s.n,
                             gpucopy_internal(s.colptr, d),
                             gpucopy_internal(s.rowval, d),
                             gpucopy_internal(s.nzval, d)))

### BASIC ARRAY OPS

for S in (:KUsparse, :Sparse)
    @eval begin
        atype{A}(::$S{A})=A
        itype{A,T,I}(::$S{A,T,I})=I
        eltype{A,T}(::$S{A,T})=T
        length(s::$S)=(s.m*s.n)
        ndims(::$S)=2
        size(s::$S)=(s.m,s.n)
        size(s::$S,i)=(i==1?s.m:i==2?s.n:error("Bad dimension"))
        strides(s::$S)=(1,s.m)
        stride(s::$S,i)=(i==1?1:i==2?s.m:error("Bad dimension"))
        isempty(s::$S)=(length(s)==0)
        to_host(s::$S{CudaArray})=cpucopy(s)
        issparse(::$S)=true
    end
end

atype(::SparseMatrixCSC)=Array
itype{T,I}(::SparseMatrixCSC{T,I})=I
full(s::KUsparse)=convert(KUdense, full(convert(SparseMatrixCSC, s)))
full{A}(s::Sparse{A})=convert(A, full(convert(SparseMatrixCSC, s)))
