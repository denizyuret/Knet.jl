# SIMILAR! create an array l.(n) similar to a given one.  If l.(n)
# exists check and resize if necessary.

function similar!(l, n, a, T=eltype(a), dims=size(a); fill=nothing)
    if !isdefined(l,n) || (typeof(l.(n)) != typeof(a))
        l.(n) = similar(a, T, dims)
        fill != nothing && fill!(l.(n), fill)
    elseif (size(l.(n)) != dims)
        l.(n) = resize!(l.(n), dims)
        fill != nothing && fill!(l.(n), fill)
    end
    return l.(n)
end

similar!(l, n, a, T, dims::Integer...; o...) = similar!(l,n,a,T,dims; o...)
similar!(l, n, a, dims::Integer...; o...) = similar!(l,n,a,dims; o...)
similar!(l, n, a, dims::Dims; o...) = similar!(l,n,a,eltype(a),dims; o...)

issimilar(a,b)=((typeof(a)==typeof(b)) && (size(a)==size(b)))
# issimilar1(a,b)=((eltype(a)==eltype(b)) && (isongpu(a)==isongpu(b)) && (length(a)==length(b)))
# issimilar2(a,b)=((eltype(a)==eltype(b)) && (isongpu(a)==isongpu(b)) && (size2(a)==size2(b)))


# This does not work in place!
# Base.resize!(a::Array, d::Dims)=similar(a, d)

# Fix bug with deepcopy, where a shared bits array is copied multiple times:

Base.deepcopy_internal{T<:Number}(x::Array{T}, s::ObjectIdDict)=(haskey(s,x)||(s[x]=copy(x));s[x])

function Base.isapprox(x, y; 
                       maxeps::Real = max(eps(eltype(x)), eps(eltype(y))),
                       rtol::Real=maxeps^(1/3), atol::Real=maxeps^(1/2))
    size(x) == size(y) || (warn("isapprox: $(size(x))!=$(size(y))"); return false)
    x = convert(Array, x)
    y = convert(Array, y)
    d = abs(x-y)
    s = abs(x)+abs(y)
    maximum(d - rtol * s) <= atol
end

Base.convert{T,I}(::Type{Array{T,2}}, a::SparseMatrixCSC{T,I})=full(a)
