# This file contains various utilities, compatibility fixes and hacks.
# Hopefully it will shrink down to nothing as things get fixed in the
# original packages.

# Print date, expression and elapsed time after execution
VERSION < v"0.4-" && eval(Expr(:using,:Dates))
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end

# Utilities
issimilar(a,b)=((typeof(a)==typeof(b)) && (size(a)==size(b)))
size2(y)=(nd=ndims(y); (nd==1 ? (length(y),1) : (stride(y, nd), size(y, nd))))
accuracy(y,z)=mean(findmax(y,1)[2] .== findmax(z,1)[2])
isongpu(a)=(GPU && isa(a, AbstractCudaArray))

# We use this function to confirm/create an array element of the right type/size
function similar!(l, n, a, T=eltype(a), dims=size(a); fill=nothing)
    if !isdefined(l,n) || (eltype(l.(n)) != T) || (size(l.(n)) != dims)
        if isa(a, AbstractSparseArray)
            l.(n) = spzeros(T, itype(a), dims...)
            fill != nothing && fill != 0 && error("Cannot fill sparse with $fill")
        elseif isa(a, DataType)
            l.(n) = a(T, dims)
            fill != nothing && fill!(l.(n), fill)
        else
            l.(n) = similar(a, T, dims)
            fill != nothing && fill!(l.(n), fill)
        end
    end
    return l.(n)
end

similar!(l, n, a, T, dims::Integer...; o...) = similar!(l,n,a,T,dims; o...)
similar!(l, n, a, dims::Integer...; o...) = similar!(l,n,a,dims; o...)
similar!(l, n, a, dims::Dims; o...) = similar!(l,n,a,eltype(a),dims; o...)

if GPU

typealias CopyableArray{T} Union(Array{T},SubArray{T},HostArray{T},CudaArray{T},CudaDynArray{T}) # no sparse

function Base.copy!{T}(dst::CopyableArray{T}, di::Integer, src::CopyableArray{T}, si::Integer, n::Integer; stream=null_stream)
    if si+n-1 > length(src) || di+n-1 > length(dst) || di < 1 || si < 1
        throw(BoundsError())
    end
    nbytes = n * sizeof(T)
    dptr = pointer(dst) + (di-1) * sizeof(T)
    sptr = pointer(src) + (si-1) * sizeof(T)
    CUDArt.rt.cudaMemcpyAsync(dptr, sptr, nbytes, CUDArt.cudamemcpykind(dst, src), stream)
    gpusync()
    return dst
end

# General cpu/gpu deep copy for composite types, gpu arrays etc.
cpucopy(x::AbstractCudaArray)=to_host(x)
cpucopy(x::AbstractArray)=(isbits(eltype(x)) ? copy(x) : map(cpucopy, x))
cpucopy(x)=mydeepcopy(x, cpucopy)
gpucopy(x::AbstractCudaArray)=copy(x)
gpucopy(x::AbstractArray)=(isbits(eltype(x)) ? CudaArray(x) : map(gpucopy, x))
gpucopy(x)=mydeepcopy(x, gpucopy)

# Adapted from deepcopy.jl:29 _deepcopy_t()
function mydeepcopy(x,fn)
    T = typeof(x)
    if T.names===() || !T.mutable || (T==Function)
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

