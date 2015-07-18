typealias BaseArray Union(Array,CudaArray)

# SIMILAR! create an array l.(n) similar to a given one.  If l.(n)
# exists check and resize if necessary.

function similar!(l, n, a, T=eltype(a), dims=size(a); fill=nothing)
    if !isdefined(l,n) || (typeof(l.(n)) != typeof(a))
        l.(n) = similar(a, T, dims)
        fill != nothing && fill!(l.(n), fill)
    elseif (size(l.(n)) != dims)
        resize!(l.(n), dims)
    end
    return l.(n)
end

similar!(l, n, a, T, dims::Integer...; o...) = similar!(l,n,a,T,dims; o...)
similar!(l, n, a, dims::Integer...; o...) = similar!(l,n,a,dims; o...)
similar!(l, n, a, dims::Dims; o...) = similar!(l,n,a,eltype(a),dims; o...)

issimilar(a,b)=((typeof(a)==typeof(b)) && (size(a)==size(b)))
# issimilar1(a,b)=((eltype(a)==eltype(b)) && (isongpu(a)==isongpu(b)) && (length(a)==length(b)))
# issimilar2(a,b)=((eltype(a)==eltype(b)) && (isongpu(a)==isongpu(b)) && (size2(a)==size2(b)))

# Here are some convenience functions for generalized columns:

csize(a)=size(a)[1:end-1]
csize(a,n)=tuple(csize(a)..., n)
clength(a)=stride(a,ndims(a))
ccount(a)=size(a,ndims(a))

size2(y)=(nd=ndims(y); (nd==1 ? (length(y),1) : (stride(y, nd), size(y, nd))))

accuracy(y,z)=mean(findmax(y,1)[2] .== findmax(z,1)[2])

atype(::Array)=Array
