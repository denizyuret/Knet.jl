import Base: dotview, view, unsafe_view, copyto!, eachindex, _maybe_reshape_parent, reshape, to_shape, SubArray, compute_stride1, axes, check_parent_index_match, IndexStyle, LinearIndices
using Base: unalias, index_ndims, @_inline_meta, @boundscheck, ViewIndex, OneTo, rdims, viewindexing, ensure_indexable, index_dimsum, fill_to_length
using Base.Broadcast: Broadcasted

### k[1:2,3:4] .= 0 => materialize!(dotview(k,1:2,3:4),broadcasted(identity,0))

# The following adapted from base: views.jl broadcast.jl subarray.jl
# Much of this will be unnecessary if we can inherit from AbstractArray

dotview(A::KnetArray,I...) = view(A,I...)

function view(A::KnetArray, I::Vararg{Any,N}) where {N}
    @_inline_meta
    J = map(i->unalias(A,i), to_indices(A, I))
    #TODO: @boundscheck checkbounds(A, J...)
    unsafe_view(_maybe_reshape_parent(A, index_ndims(J...)), J...)
end

function unsafe_view(A::KnetArray, I::Vararg{ViewIndex,N}) where {N}
    @_inline_meta
    SubArray(A, I)
end

eachindex(::IndexLinear, A::KnetArray) = (@_inline_meta; OneTo(length(A)))
LinearIndices(A::KnetArray) = LinearIndices(axes(A))

_maybe_reshape_parent(A::KnetArray, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::KnetArray{<:Any,1}, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::KnetArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A
_maybe_reshape_parent(A::KnetArray, ::NTuple{N, Bool}) where {N} = reshape(A, Val(N))

reshape(parent::KnetArray{T,N}, ndims::Val{N}) where {T,N} = parent
reshape(parent::KnetArray, ndims::Val{N}) where N = reshape(parent, rdims(Val(N), axes(parent)))
reshape(parent::KnetArray, shp::Tuple{Union{Integer,OneTo}, Vararg{Union{Integer,OneTo}}}) = reshape(parent, to_shape(shp))

function SubArray(parent::KnetArray, indices::Tuple)
    @_inline_meta
    SubArray(IndexStyle(viewindexing(indices), IndexStyle(parent)), parent, ensure_indexable(indices), index_dimsum(indices...))
end

IndexStyle(::KnetArray)=IndexLinear()

compute_stride1(parent::KnetArray, I::NTuple{N,Any}) where {N} =
    (@_inline_meta; compute_stride1(1, fill_to_length(axes(parent), OneTo(1), Val(N)), I))

function axes(A::KnetArray{T,N}, d) where {T,N}
    @_inline_meta
    d <= N ? axes(A)[d] : OneTo(1)
end

check_parent_index_match(parent::KnetArray{T,N}, ::NTuple{N, Bool}) where {T,N} = nothing

# dotview(P::KnetArray{T,N},I...) where {T,N} =SubArray{T,N,typeof(P),typeof(I),false}(P,I,0,0)
# check_parent_index_match(parent::KnetArray{T,N}, ::NTuple{N, Bool}) where {T,N} = nothing
