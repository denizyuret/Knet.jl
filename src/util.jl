# This file contains various utilities, compatibility fixes and hacks.
# Hopefully it will shrink down to nothing as things get fixed in the
# original packages.

# Print date, expression and elapsed time after execution
VERSION < v"0.4-" && eval(Expr(:using,:Dates))
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end

issimilar(a,b)=((typeof(a)==typeof(b)) && (size(a)==size(b)))
size2(y)=(nd=ndims(y); (nd==1 ? (length(y),1) : (stride(y, nd), size(y, nd))))
accuracy(y,z)=mean(findmax(y,1)[2] .== findmax(z,1)[2])
isongpu(a)=(GPU && isa(a, AbstractCudaArray))
itype{Tv,Ti}(::SparseMatrixCSC{Tv,Ti})=Ti
Base.similar{Tv,Ti}(::SparseMatrixCSC{Tv,Ti},m,n)=spzeros(Tv,Ti,m,n)
similar!(l,n,a,d::Integer...; o...)=similar!(l,n,a,d;o...)

function similar!(l, n, a, dims=size(a); fill=nothing)
    if !isdefined(l,n) || (size(l.(n)) != dims)
        if isa(a, AbstractSparseArray)
            l.(n) = spzeros(eltype(a), itype(a), dims...)
            fill != nothing && fill != 0 && error("Cannot fill sparse with $fill")
        else
            l.(n) = similar(a, dims)
            fill != nothing && fill!(l.(n), fill)
        end
    end
    return l.(n)
end

if GPU   ########## CUDA extensions:

const libkunet = find_library(["libkunet"], [Pkg.dir("KUnet/src")])

convert{T,S}(::Type{CudaArray{T}}, x::Array{S})=CudaArray(convert(Array{T}, x))
convert{T,S}(::Type{Array{T}}, x::CudaArray{S})=convert(Array{T}, to_host(x))
reshape(a::CudaArray, dims::Dims)=reinterpret(eltype(a), a, dims)
reshape(a::CudaArray, dims::Int...)=reshape(a, dims)
rand!(A::CudaArray{Float32})=(ccall((:randfill32,libkunet),Void,(Cint,Ptr{Float32}),length(A),A); A)
rand!(A::CudaArray{Float64})=(ccall((:randfill64,libkunet),Void,(Cint,Ptr{Float64}),length(A),A); A)
fill!(A::CudaArray,x::Number)=cudnnSetTensor(A, x)


typealias CopyableArray{T} Union(Array{T},SubArray{T},HostArray{T},CudaArray{T})

function copy!{T}(dst::CopyableArray{T}, di::Integer, src::CopyableArray{T}, si::Integer, n::Integer; stream=null_stream)
    if si+n-1 > length(src) || di+n-1 > length(dst) || di < 1 || si < 1
        throw(BoundsError())
    end
    nbytes = n * sizeof(T)
    dptr = pointer(dst) + (di-1) * sizeof(T)
    sptr = pointer(src) + (si-1) * sizeof(T)
    CUDArt.rt.cudaMemcpyAsync(dptr, sptr, nbytes, CUDArt.cudamemcpykind(dst, src), stream)
    return dst
end

# For debugging
function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end

# our version of srand sets both gpu and cpu
function gpuseed(n::Integer)
    srand(n)
    GPU && ccall((:gpuseed,libkunet),Void,(Culonglong,),convert(Culonglong, n))
end

# matmul.jl:116
function (*){T}(A::CudaMatrix{T}, B::CudaMatrix{T})
    C = similar(A,T,(size(A,1),size(B,2)))
    gemm!('N','N',one(T),A,B,zero(T),C)
end

# without this patch, deepcopy does not work on structs with CudaArrays
function Base.deepcopy_internal(x::CudaArray, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    copy(x)
end


cpucopy(x::CudaArray)=to_host(x)
cpucopy(x::AbstractArray)=(isbits(eltype(x)) ? copy(x) : map(cpucopy, x))
cpucopy(x)=mydeepcopy(x, cpucopy)
gpucopy(x::CudaArray)=copy(x)
gpucopy(x::AbstractArray)=(isbits(eltype(x)) ? CudaArray(x) : map(gpucopy, x))
gpucopy(x)=mydeepcopy(x, gpucopy)

# Adapted from deepcopy.jl:29 _deepcopy_t()
function mydeepcopy(x,fn)
    T = typeof(x)
    if T.names===() || !T.mutable
        return x
    end
    ret = ccall(:jl_new_struct_uninit, Any, (Any,), T)
    for i in 1:length(T.names)
        if isdefined(x,i)
            ret.(i) = fn(x.(i))
        end
    end
    return ret
end

else  # if GPU

# Need this so code works without gpu
cpucopy(x)=deepcopy(x)
gpucopy(x)=error("No GPU")

end   # if GPU

