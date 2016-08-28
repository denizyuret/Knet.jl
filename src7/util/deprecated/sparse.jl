using CUDArt
import Base: isequal, convert, similar, copy, copy!, eltype, length, ndims, size, isempty, issparse, stride, strides, full
import CUDArt: to_host

# I want to make the base array explicit in the type signature of
# sparse arrays: So instead of SparseMatrixCSC{T,Int32} we use the
# equivalent KUsparse{Array,T}. Sparse arrays on the gpu can then be
# represented with KUsparse{CudaArray,T}.  The index type is always
# Int32, unlike SparseMatrixCSC, KUsparse does not support Int64
# indices.

type KUsparse{A,T}
    m                       # Number of rows
    n                       # Number of columns
    colptr                  # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval                  # Row values of nonzeros
    nzval                   # Nonzero values
end

# construct KUsparse
KUsparse{A<:Array,T}(::Type{A}, ::Type{T}, m::Integer, n::Integer)=KUsparse{A,T}(m,n,ones(Int32,n+1),Array(Int32,0),Array(T,0))
KUsparse{A<:CudaArray,T}(::Type{A}, ::Type{T}, m::Integer, n::Integer)=KUsparse{A,T}(m,n,CudaArray(ones(Int32,n+1)),CudaArray(Int32,0),CudaArray(T,0))
KUsparse{A,T}(::Type{A}, ::Type{T}, d::NTuple{2,Int})=KUsparse(A,T,d...)
similar{A,T}(s::KUsparse{A}, ::Type{T}, m::Integer, n::Integer)=KUsparse(A,T,m,n)
similar{A,T}(s::KUsparse{A}, ::Type{T}, d::NTuple{2,Int})=KUsparse(A,T,d...)

isequal(a::KUsparse,b::KUsparse)=((typeof(a)==typeof(b)) && (sizeof(a)==sizeof(b)) && isequal(a.colptr,b.colptr) && isequal(a.rowval,b.rowval) && isequal(a.nzval,b.nzval))

# convert to KUsparse
convert{A<:Array,T}(::Type{KUsparse{A,T}}, s::SparseMatrixCSC{T,Int32})=KUsparse{Array,T}(s.m,s.n,s.colptr,s.rowval,s.nzval)
convert{A<:CudaArray,T}(::Type{KUsparse{A,T}}, s::SparseMatrixCSC{T,Int32})=KUsparse{CudaArray,T}(s.m,s.n,CudaArray(s.colptr),CudaArray(s.rowval),CudaArray(s.nzval))
convert{A<:BaseArray,T}(::Type{KUsparse{A,T}}, s::AbstractArray{T})=convert(KUsparse{A,T}, convert(SparseMatrixCSC{T,Int32}, s))
convert{A<:BaseArray,T}(::Type{KUsparse{A}}, s::AbstractArray{T})=convert(KUsparse{A,T}, convert(SparseMatrixCSC{T,Int32}, s))
convert{T}(::Type{KUsparse}, s::AbstractArray{T})=convert(KUsparse{Array,T}, s)
convert{T}(::Type{KUsparse}, s::AbstractCudaArray{T})=convert(KUsparse{CudaArray,T}, s)

# convert to SparseMatrixCSC
convert{A<:SparseMatrixCSC}(::Type{A}, s::KUsparse)=SparseMatrixCSC(s.m,s.n,convert(Array,s.colptr),convert(Array,s.rowval),convert(Array,s.nzval))
convert{A<:SparseMatrixCSC}(::Type{A}, a::CudaMatrix)=convert(SparseMatrixCSC, to_host(a))

# convert to Array/CudaArray
convert{A<:Array}(::Type{A}, s::SparseMatrixCSC)=full(s)
convert{A<:CudaArray}(::Type{A}, s::SparseMatrixCSC)=CudaArray(convert(Array, s))
convert{A<:Array}(::Type{A}, s::KUsparse)=convert(Array, convert(SparseMatrixCSC, s))
convert{A<:CudaArray}(::Type{A}, s::KUsparse)=convert(CudaArray, convert(SparseMatrixCSC, s))

# Now we can construct a KUsparse{CudaArray,T} using gpucopy:

copy(a::KUsparse)=deepcopy(a)
# copy{A<:BaseArray,T}(a::KUsparse{A,T})=KUsparse{A,T}(a.m,a.n,copy(a.colptr),copy(a.rowval),copy(a.nzval))

cpucopy_internal{A<:CudaArray,T}(s::KUsparse{A,T},d::ObjectIdDict)=
    (haskey(d,s)||(d[s] = KUsparse{Array,T}(s.m, s.n, cpucopy_internal(s.colptr,d), cpucopy_internal(s.rowval,d), cpucopy_internal(s.nzval,d))); d[s])
                       
gpucopy_internal{A<:Array,T}(s::KUsparse{A,T},d::ObjectIdDict)=
    (haskey(d,s)||(d[s] = KUsparse{CudaArray,T}(s.m, s.n, gpucopy_internal(s.colptr,d), gpucopy_internal(s.rowval,d), gpucopy_internal(s.nzval,d))); d[s])

atype{A}(::KUsparse{A})=A
itype(::KUsparse)=Int32
eltype{A,T}(::KUsparse{A,T})=T
length(s::KUsparse)=(s.m*s.n)
ndims(::KUsparse)=2
size(s::KUsparse)=(s.m,s.n)
size(s::KUsparse,i)=(i==1?s.m:i==2?s.n:error("Bad dimension"))
strides(s::KUsparse)=(1,s.m)
stride(s::KUsparse,i)=(i==1?1:i==2?s.m:error("Bad dimension"))
isempty(s::KUsparse)=(length(s)==0)
to_host(s::KUsparse{CudaArray})=cpucopy(s)
issparse(::KUsparse)=true
# This won't work with CudaArray
# nnz(s::KUsparse)=s.colptr[s.n+1]-1  # TODO: shall we keep colptr in RAM?

atype(::SparseMatrixCSC)=Array
itype{T,I}(::SparseMatrixCSC{T,I})=I
full(s::KUsparse)=convert(KUdense, full(convert(SparseMatrixCSC, s)))
copy!{A,T}(a::KUsparse{A,T}, b::SparseMatrixCSC{T,Int64})=copy!(a, convert(SparseMatrixCSC{T,Int32},b))

function copy!{A,T}(a::KUsparse{A,T}, b::SparseMatrixCSC{T,Int32})
    a.m=b.m
    a.n=b.n
    for n in (:colptr, :rowval, :nzval)
        length(a.(n)) != length(b.(n)) && resize!(a.(n), length(b.(n)))
        copy!(a.(n), b.(n))
    end
    return a
end

# The final prediction output y should match the input x as closely as
# possible except for being dense.  These functions support obtaining
# the dense version of x.

dtype(x)=typeof(x)
dtype{T}(x::SparseMatrixCSC{T})=Array{T}
dtype{A,T}(x::KUsparse{A,T})=KUdense{A,T}
dtype{A,T}(x::KUdense{A,T})=KUdense{A,T}  # this is necessary, otherwise 1-dim x does not match 2-dim y.

dsimilar(x,d::Dims)=similar(x,d)
dsimilar{T}(x::SparseMatrixCSC{T},d::Dims)=Array(T,d)
dsimilar{A,T}(x::KUsparse{A,T},d::Dims)=KUdense(A,T,d)

function dsimilar!(l, n, x, dims=size(x))
    if (!isdefined(l,n) || !isa(l.(n), dtype(x)))
        l.(n) = dsimilar(x, dims)
    elseif (size(l.(n)) != dims)
        l.(n) = resize!(l.(n), dims)
    end
    return l.(n)
end

# type Sparse{A,T,I<:Integer}; m; n; colptr; rowval; nzval; end

# convert{T,I}(::Type{Sparse}, s::SparseMatrixCSC{T,I})=convert(Sparse{Array}, s)

# convert{A,T,I}(::Type{Sparse{A}}, s::SparseMatrixCSC{T,I})=
#     Sparse{A,T,I}(s.m,s.n,
#                   convert(A, s.colptr),
#                   convert(A, s.rowval),
#                   convert(A, s.nzval))

# convert{A,T,I}(::Type{SparseMatrixCSC}, s::Sparse{A,T,I})=
#     SparseMatrixCSC(s.m,s.n,
#                     convert(Array, s.colptr),
#                     convert(Array, s.rowval),
#                     convert(Array, s.nzval))

# convert{T}(::Type{Sparse}, a::Array{T,2})=convert(Sparse{Array}, a)
# convert{T}(::Type{Sparse}, a::CudaArray{T,2})=convert(Sparse{CudaArray}, a)

# convert{A<:Array,T}(::Type{Sparse{A}}, a::BaseArray{T,2})=
#     convert(Sparse, convert(SparseMatrixCSC{T,Int32}, sparse(convert(Array, a))))

# convert{A<:CudaArray,T}(::Type{Sparse{A}}, a::BaseArray{T,2})=
#     gpucopy(convert(Sparse{Array}, a))

# # Now we can construct a Sparse{CudaArray,T,I} using gpucopy:

# cpucopy_internal{A<:CudaArray,T,I}(s::Sparse{A,T,I},d::ObjectIdDict)=
#     (haskey(d,s) ? d[s] : 
#      Sparse{Array,T,I}(s.m, s.n,
#                        cpucopy_internal(s.colptr,d),
#                        cpucopy_internal(s.rowval,d),
#                        cpucopy_internal(s.nzval,d)))

# gpucopy_internal{A<:Array,T,I}(s::Sparse{A,T,I},d::ObjectIdDict)=
#     (haskey(d,s) ? d[s] : 
#      Sparse{CudaArray,T,I}(s.m,s.n,
#                            gpucopy_internal(s.colptr,d),
#                            gpucopy_internal(s.rowval,d),
#                            gpucopy_internal(s.nzval,d)))

# # And we can construct KUsparse which uses resizeable KUdense arrays for members:

# type KUsparse0{A,T,I<:Integer}
#     m::Int                      # Number of rows
#     n::Int                      # Number of columns
#     colptr::KUdense{A,I,1}      # Column i is in colptr[i]+1:colptr[i+1], note that this is 0 based on cusparse
#     rowval::KUdense{A,I,1}      # Row values of nonzeros
#     nzval::KUdense{A,T,1}       # Nonzero values
# end

# KUsparse0{A<:Array,T,I}(::Type{A}, ::Type{T}, ::Type{I}, m::Integer, n::Integer)=
#     KUsparse0{A,T,I}(m,n,KUdense(ones(I,n+1)),KUdense(Array(I,0)),KUdense(Array(T,0)))

# KUsparse0{A<:CudaArray,T,I}(::Type{A}, ::Type{T}, ::Type{I}, m::Integer, n::Integer)=
#     KUsparse0{A,T,I}(m,n,KUdense(CudaArray(ones(I,n+1))),KUdense(CudaArray(I,0)),KUdense(CudaArray(T,0)))

# KUsparse0{A,T,I}(::Type{A}, ::Type{T}, ::Type{I}, d::NTuple{2,Int})=KUsparse0(A,T,I,d...)

# similar{A,T,I}(s::KUsparse0{A}, ::Type{T}, ::Type{I}, m::Integer, n::Integer)=KUsparse0(A,T,I,m,n)

# similar{A,T,I,U}(s::KUsparse0{A,T,I}, ::Type{U}, d::NTuple{2,Int})=KUsparse0(A,U,I,d...)

# convert{A,T,I}(::Type{KUsparse0}, s::Sparse{A,T,I})=
#     KUsparse0{A,T,I}(s.m,s.n,convert(KUdense,s.colptr),convert(KUdense,s.rowval),convert(KUdense,s.nzval))

# convert{T,I}(::Type{KUsparse0}, s::SparseMatrixCSC{T,I})=
#     KUsparse0{Array,T,I}(s.m,s.n,convert(KUdense,s.colptr),convert(KUdense,s.rowval),convert(KUdense,s.nzval))

# convert{A,T,I}(::Type{Sparse}, s::KUsparse0{A,T,I})=
#     Sparse{A,T,I}(s.m,s.n,convert(A,s.colptr),convert(A,s.rowval),convert(A,s.nzval))

# convert{A,T,I}(::Type{SparseMatrixCSC}, s::KUsparse0{A,T,I})=
#     SparseMatrixCSC(s.m,s.n,convert(Array,s.colptr),convert(Array,s.rowval),convert(Array,s.nzval))

# convert{A<:KUsparse0{Array},T}(::Type{A}, a::BaseArray{T,2})=
#     convert(KUsparse0, convert(SparseMatrixCSC{T,Int32}, sparse(convert(Array, a))))

# convert{A<:KUsparse0{CudaArray},T}(::Type{A}, a::BaseArray{T,2})=
#     gpucopy(convert(KUsparse0{Array}, a))

# convert{A<:KUsparse0{CudaArray},T,I}(::Type{A}, s::SparseMatrixCSC{T,I})=
#     KUsparse0{CudaArray,T,I}(s.m,s.n,KUdense(CudaArray(s.colptr)),KUdense(CudaArray(s.rowval)),KUdense(CudaArray(s.nzval)))

# convert{A<:KUsparse0{Array},T,I}(::Type{A}, s::SparseMatrixCSC{T,I})=
#     KUsparse0{Array,T,I}(s.m,s.n,KUdense(s.colptr),KUdense(s.rowval),KUdense(s.nzval))

# ### BASIC COPY

# copy!{A,B,T,I}(a::KUsparse0{A,T,I}, b::KUsparse0{B,T,I})=
#     (a.m=b.m;a.n=b.n;for f in (:colptr,:rowval,:nzval); copy!(a.(f),b.(f)); end; a)

# copy!{A,T,I}(a::KUsparse0{A,T,I}, b::SparseMatrixCSC{T,I})=
#     (a.m=b.m;a.n=b.n;for f in (:colptr,:rowval,:nzval); copy!(a.(f),b.(f)); end; a)

# copy(a::KUsparse0)=KUsparse0(a.m,a.n,copy(a.colptr),copy(a.rowval),copy(a.nzval))


# # We need to fix cpu/gpu copy so the type changes appropriately:

# cpucopy_internal{A<:CudaArray,T,I}(s::KUsparse0{A,T,I},d::ObjectIdDict)=
#     (haskey(d,s) ? d[s] :
#      KUsparse0{Array,T,I}(s.m,s.n,
#                          cpucopy_internal(s.colptr, d),
#                          cpucopy_internal(s.rowval, d),
#                          cpucopy_internal(s.nzval, d)))

# gpucopy_internal{A<:Array,T,I}(s::KUsparse0{A,T,I},d::ObjectIdDict)=
#     (haskey(d,s) ? d[s] : 
#      KUsparse0{CudaArray,T,I}(s.m,s.n,
#                              gpucopy_internal(s.colptr, d),
#                              gpucopy_internal(s.rowval, d),
#                              gpucopy_internal(s.nzval, d)))

# ### BASIC ARRAY OPS

# for S in (:KUsparse0, :Sparse)
#     @eval begin
#         atype{A}(::$S{A})=A
#         itype{A,T,I}(::$S{A,T,I})=I
#         eltype{A,T}(::$S{A,T})=T
#         length(s::$S)=(s.m*s.n)
#         ndims(::$S)=2
#         size(s::$S)=(s.m,s.n)
#         size(s::$S,i)=(i==1?s.m:i==2?s.n:error("Bad dimension"))
#         strides(s::$S)=(1,s.m)
#         stride(s::$S,i)=(i==1?1:i==2?s.m:error("Bad dimension"))
#         isempty(s::$S)=(length(s)==0)
#         to_host(s::$S{CudaArray})=cpucopy(s)
#         issparse(::$S)=true
#     end
# end

# atype(::SparseMatrixCSC)=Array
# itype{T,I}(::SparseMatrixCSC{T,I})=I
# full(s::KUsparse0)=convert(KUdense, full(convert(SparseMatrixCSC, s)))
# full{A}(s::Sparse{A})=convert(A, full(convert(SparseMatrixCSC, s)))

# convert{A<:CudaArray,T,I}(::Type{A}, s::SparseMatrixCSC{T,I})=CudaArray(full(s))
# convert{A<:SparseMatrixCSC,T,N}(::Type{A}, a::CudaArray{T,N})=sparse(to_host(a))

# convert{A<:Array,B,T,I}(::Type{A}, s::KUsparse0{B,T,I})=convert(Array, convert(SparseMatrixCSC, s))
# convert{A<:CudaArray,B,T,I}(::Type{A}, s::KUsparse0{B,T,I})=convert(CudaArray, convert(SparseMatrixCSC, s))

# # These two already defined in sparsematrix.jl:
# # convert(::Type{Matrix}, S::SparseMatrixCSC) = full(S)
# # sparse{Tv}(A::Matrix{Tv}) = convert(SparseMatrixCSC{Tv,Int}, A)
# # But they don't cover general array conversion:
# convert(::Type{Array}, S::SparseMatrixCSC) = full(S)