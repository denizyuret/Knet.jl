# TODO: put all this into a new DynamicArray library.
# what is wrong with DynamicDenseCPU and DynamicDenseGPU?
# prevent resize from changing ndims

using CUDArt
import Base: isequal, similar, convert, copy, copy!, resize!, issparse
import Base: rand!, randn!, fill!
import CUDArt: to_host

### KUdense parametrized by array type, element type, and ndims:
# TODO: we should get rid of N, because it may change during a resize!

type KUdense{A,T,N}; arr; ptr; end

### CONSTRUCTORS

KUdense(a)=KUdense{atype(a),eltype(a),ndims(a)}(a, reshape(a, length(a)))

KUdense{A<:Array,T,N}(::Type{A}, ::Type{T}, d::NTuple{N,Int})=KUdense(Array(T,d))
KUdense{A<:CudaArray,T,N}(::Type{A}, ::Type{T}, d::NTuple{N,Int})=KUdense(CudaArray(T,d))

convert(::Type{KUdense}, a)=KUdense(a)
convert{A<:BaseArray}(::Type{A}, a::KUdense)=convert(A, a.arr)
convert{A,B}(::Type{KUdense{B}}, a::A)=KUdense(convert(B, a))
convert{A<:Array,B<:CudaArray,T,N}(::Type{KUdense{A,T,N}}, a::KUdense{B,T,N})=cpucopy(a)
convert{A<:CudaArray,B<:Array,T,N}(::Type{KUdense{A,T,N}}, a::KUdense{B,T,N})=gpucopy(a)

similar{A,T,N}(a::KUdense{A}, ::Type{T}, d::NTuple{N,Int})=KUdense(A,T,d)
similar{A,T,N}(a::KUdense{A,T}, d::NTuple{N,Int})=KUdense(A,T,d)
similar{A,T,N}(a::KUdense{A,T,N})=KUdense(A,T,size(a))

arr(a::Vector,d::Dims)=pointer_to_array(pointer(a), d)
arr(a::CudaVector,d::Dims)=CudaArray(a.ptr, d, a.dev)
arr(a)=(isa(a,KUdense) ? a.arr : a)

atype(::Array)=Array
atype(::SubArray)=Array
atype(::CudaArray)=CudaArray

isequal(a::KUdense,b::KUdense)=((typeof(a)==typeof(b)) && (sizeof(a)==sizeof(b)) && isequal(a.arr,b.arr))

### BASIC ARRAY OPS

atype{A}(::KUdense{A})=A

for fname in (:eltype, :length, :ndims, :size, :strides, :pointer, :isempty, :vecnorm)
    @eval (Base.$fname)(a::KUdense)=$fname(a.arr)
end

for fname in (:size, :stride)
    @eval (Base.$fname)(a::KUdense,n)=$fname(a.arr,n)
end

for fname in (:getindex, :setindex!, :sub)
    @eval (Base.$fname)(a::KUdense,n...)=$fname(a.arr,n...)
end

### BASIC COPY

copy!{A,B,T}(a::KUdense{A,T}, b::KUdense{B,T})=(resize!(a, size(b)); copy!(a.arr, 1, b.arr, 1, length(b)); a)
copy!{A,T}(a::KUdense{A,T}, b::Union{Array{T},CudaArray{T}})=(resize!(a, size(b)); copy!(a.arr, 1, b, 1, length(b)); a)
copy!{A,T}(a::Union{Array{T},CudaArray{T}}, b::KUdense{A,T})=(size(a)==size(b)||error(); copy!(a, 1, b.arr, 1, length(b)); a)
copy!{A,B,T}(a::KUdense{A,T}, ai::Integer, b::KUdense{B,T}, bi::Integer, n::Integer)=(copy!(a.arr, ai, b.arr, bi, n); a)
copy{A,T,N}(a::KUdense{A,T,N})=copy!(similar(a), a)

### EFFICIENT RESIZE

# Resize factor: 1.3 ensures a3 can be written where a0+a1 used to be
resizefactor(::Type{KUdense})=1.3

function resize!(a::KUdense, d::Dims)
    size(a)==d && return a
    n = prod(d)
    n > length(a.ptr) && resize!(a.ptr, round(Int,resizefactor(KUdense)*n+1))
    a.arr = arr(a.ptr, d)
    return a
end

resize!(a::KUdense, d::Int...)=resize!(a,d)

# Need to fix deepcopy so it does not create two arrays for arr and ptr:
# And atype changes correctly.

deepcopy_internal(x::KUdense,s::ObjectIdDict)=(haskey(s,x)||(s[x]=copy(x));s[x])
cpucopy_internal(x::KUdense,s::ObjectIdDict)=deepcopy_internal(x,s)
gpucopy_internal(x::KUdense,s::ObjectIdDict)=deepcopy_internal(x,s)

cpucopy_internal{A<:CudaArray}(x::KUdense{A},s::ObjectIdDict)=(haskey(s,x)||(s[x]=KUdense(to_host(x.arr)));s[x])
gpucopy_internal{A<:Array}(x::KUdense{A},s::ObjectIdDict)=(haskey(s,x)||(s[x]=KUdense(CudaArray(x.arr)));s[x])

randn!{A,T}(a::KUdense{A,T}, mean=zero(T), std=one(T))=(randn!(a.arr, mean, std); a)
rand!(a::KUdense)=(rand!(a.arr); a)
fill!{A,T}(a::KUdense{A,T},x)=(fill!(a.arr,convert(T,x)); a)

to_host{A<:CudaArray}(a::KUdense{A})=cpucopy(a)
issparse(a::KUdense)=false

# The final prediction output y should match the input x as closely as
# possible except for being dense.  These functions support obtaining
# the dense version of x.

dtype(x)=typeof(x)
dtype{T}(x::SparseMatrixCSC{T})=Array{T}
dtype{T}(x::CudaSparseMatrixCSC{T})=KUdense{CudaArray,T} # TODO: should this be CudaArray?
dtype{A,T}(x::KUdense{A,T})=KUdense{A,T}  # this is necessary, otherwise 1-dim x does not match 2-dim y.

dsimilar(x,d::Dims)=similar(x,d)
dsimilar{T}(x::SparseMatrixCSC{T},d::Dims)=Array(T,d)
dsimilar{T}(x::CudaSparseMatrixCSC{T},d::Dims)=KUdense(CudaArray,T,d)

function dsimilar!(l, n, x, dims=size(x))
    if (!isdefined(l,n) || !isa(l.(n), dtype(x)))
        l.(n) = dsimilar(x, dims)
    elseif (size(l.(n)) != dims)
        l.(n) = resize!(l.(n), dims)
    end
    return l.(n)
end

