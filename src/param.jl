import Base: isequal, convert

type KUparam{A,T,N}; arr; diff; init; initp; lr; gc; l1reg; l2reg; adagrad; ada; momentum; mom; nesterov; nes; average; avg; inc; KUparam()=new(); end

KUparam(w; o...)=init(setopt!(KUparam{atype(w),eltype(w),ndims(w)}(); arr=w, o...))
KUparam{A,T}(::Type{A}, ::Type{T}, d::Dims; o...)=KUparam(A(T,d); o...)
KUparam{A,T}(::Type{A}, ::Type{T}, d1::Int, d::Int...; o...)=KUparam(A,T,tuple(d1,d...); o...)
KUparam{T}(::Type{T}, d::Dims; o...)=KUparam((gpu()?CudaArray:Array),T,d; o...)
KUparam{T}(::Type{T}, d::Int...; o...)=KUparam(T, d; o...)
KUparam(d::Int...; o...)=KUparam(Float64, d; o...)
setopt!(p::KUparam; o...)=(for (n,v) in o; p.(n)=v; end; p)

# TODO: create an rgen type: makes things serializable
function init(p::KUparam, T::DataType=eltype(p), d::Dims=size(p.arr))
    (size(p.arr)==d && eltype(p.arr)==T) || (p.arr = similar(p.arr, T, d))
    # we want no init if params given in matrix
    if !isempty(p.arr) && nz(p,:init,nothing)
        initp = (!isdefined(p,:initp) ? () :
                 isa(p.initp,Tuple) ? p.initp :
                 (p.initp,))
        p.init(p.arr, initp...)
    end
    return p
end

function isequal(a::KUparam,b::KUparam)
    typeof(a) == typeof(b) || return false
    size(a) == size(b) || return false
    for n in fieldnames(a)
        if isdefined(a,n) && isdefined(b,n)
            isequal(a.(n), b.(n)) || return false
        elseif isdefined(a,n) || isdefined(b,n)
            return false
        end
    end
    return true
end

# BASIC ARRAY OPS:

for fname in (:eltype, :length, :ndims, :size, :strides, :pointer, :isempty, :vecnorm)
    @eval (Base.$fname)(a::KUparam)=$fname(a.arr)
end

for fname in (:size, :stride)
    @eval (Base.$fname)(a::KUparam,n)=$fname(a.arr,n)
end

atype{A}(::KUparam{A})=A
diff(a::KUparam)=a.diff
difnorm(a::KUparam)=(isdefined(a,:diff) ? vecnorm(a.diff) : 0)

# DEPRECATED:
# update!(::Nothing;o...)=nothing
# setopt!(::Nothing;o...)=nothing
# initdiff(w::KUparam; fill=nothing, o...)=(similar!(w, :diff, w.arr); fill!=nothing && fill!(w.diff,fill); w)

# We need to fix cpu/gpu copy so the type changes appropriately:

copy(x::KUparam)=deepcopy(x)

function cpucopy_internal{A<:CudaArray,T,N}(x::KUparam{A,T,N},d::ObjectIdDict)
    haskey(d,x) && return d[x]
    y = KUparam{Array,T,N}()
    for n in fieldnames(x)
        isdefined(x,n) || continue
        y.(n) = cpucopy_internal(x.(n),d)
    end
    d[x] = y
end

function gpucopy_internal{A<:Array,T,N}(x::KUparam{A,T,N},d::ObjectIdDict)
    haskey(d,x) && return d[x]
    y = KUparam{CudaArray,T,N}()
    for n in fieldnames(x)
        isdefined(x,n) || continue
        y.(n) = gpucopy_internal(x.(n),d)
    end
    d[x] = y
end

convert{A<:BaseArray}(::Type{A}, a::KUparam)=convert(A, a.arr)
convert{A<:BaseArray}(::Type{KUparam}, a::A)=KUparam(a)

# TODO: both weights and training parameters are called param; setopt! vs params is confusing
# Probably should rename this back to Param after moving others to DynamicArray.
# How does caffe deal with this problem?
