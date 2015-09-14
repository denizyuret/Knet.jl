# SIMILAR! create an array l.(n) similar to a given one.  If l.(n)
# exists check and resize if necessary.

function similar!(l, n, a, T=eltype(a), dims=size(a); fill=nothing)
    if !isdefined(l,n) || (typeof(l.(n)) != typeof(a))
        l.(n) = similar(a, T, dims)
        fill != nothing && fill!(l.(n), fill)
    elseif (size(l.(n)) != dims)
        l.(n) = resize!(l.(n), dims)
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
# We consider a 1-D array a single column:

csize(a)=(ndims(a)==1 ? size(a) : size(a)[1:end-1])
csize(a,n)=tuple(csize(a)..., n) # size if you had n columns
clength(a)=(ndims(a)==1 ? length(a) : stride(a,ndims(a)))
ccount(a)=(ndims(a)==1 ? 1 : size(a,ndims(a)))
size2(y)=(nd=ndims(y); (nd==1 ? (length(y),1) : (stride(y, nd), size(y, nd)))) # size as a matrix

# This does not work in place!
# Base.resize!(a::Array, d::Dims)=similar(a, d)

# Fix bug with deepcopy, where a shared bits array is copied multiple times:

Base.deepcopy_internal{T<:Number}(x::Array{T}, s::ObjectIdDict)=(haskey(s,x)||(s[x]=copy(x));s[x])
