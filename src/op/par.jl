# TODO: update
# TODO: averaging

type Par <: Op; dims; init; initialized; out0; out; dif;
    lr; l1reg; l2reg; adagrad; ada; momentum; mom; nesterov; nes; average; avg; 
    Par()=new(); 
end

setopt!(p::Par; o...)=(for (n,v) in o; p.(n)=v; end; p)
par(; o...)=setopt!(Par(); initialized=false, o...)
par(i::Integer, d::Integer...; o...)=par(; dims=(i,d...), o...)
par(w::AbstractArray; o...)=par(; out0=w, dims=size(w), o...)

infersize(p::Par)=(isdefined(p,:dims) ? (p.dims,) : nothing)
ninputs(::Par)=0
overwrites(::Par)=false
back_reads_x(::Par)=false
back_reads_y(::Par)=false

abstract Rgen
type Gaussian <: Rgen; mean; std; end
type Uniform  <: Rgen; min; max; end
type Constant <: Rgen; val; end
type Identity <: Rgen; val; Identity(x=1)=new(x); end

function back(p::Par, dy; y=nothing, o...)
    isdefined(p, :dif) || (p.dif = dy)
    @assert dy === p.dif
    @assert y === p.out
end

function forw(p::Par, y; o...)
    if p.initialized
        @assert p.out === y "p.out=$(p.out) y=$y"
        return y
    else
        p.initialized = true
        p.out = 
        (isdefined(p, :out0)   ? copy!(y, p.out0) :
         !isdefined(p, :init)  ? scale!(0.01, randn!(y)) :
         isa(p.init, Constant) ? fill!(y, p.init.val) :
         isa(p.init, Uniform)  ? (rand!(y); axpb!(p.init.max - p.init.min, p.init.min, y)) :
         isa(p.init, Gaussian) ? (randn!(y); axpb!(p.init.std, p.init.mean, y)) : 
         isa(p.init, Identity) ? scale!(p.init.val, copy!(y, eye(eltype(y), size(y)...))) :
         error())
    end
end

function Base.isequal(a::Par,b::Par)
    for n in fieldnames(a)
        if isdefined(a,n) && isdefined(b,n)
            isequal(a.(n), b.(n)) || return false
        elseif isdefined(a,n) || isdefined(b,n)
            return false
        end
    end
    return true
end



### DEAD CODE

# BASIC ARRAY OPS:

# for fname in (:eltype, :length, :ndims, :size, :strides, :pointer, :isempty, :vecnorm)
#     @eval (Base.$fname)(a::Par)=$fname(a.arr)
# end

# for fname in (:size, :stride)
#     @eval (Base.$fname)(a::Par,n)=$fname(a.arr,n)
# end

# # atype{A}(::Par{A})=A
# diff(a::Par)=a.diff
# difnorm(a::Par)=(isdefined(a,:diff) ? vecnorm(a.diff) : 0)

# DEPRECATED:
# update!(::Nothing;o...)=nothing
# setopt!(::Nothing;o...)=nothing
# initdiff(w::Par; fill=nothing, o...)=(similar!(w, :diff, w.arr); fill!=nothing && fill!(w.diff,fill); w)

# We need to fix cpu/gpu copy so the type changes appropriately:

# copy(x::Par)=deepcopy(x)

# function cpucopy_internal{A<:CudaArray,T,N}(x::Par{A,T,N},d::ObjectIdDict)
#     haskey(d,x) && return d[x]
#     y = Par{Array,T,N}()
#     for n in fieldnames(x)
#         isdefined(x,n) || continue
#         y.(n) = cpucopy_internal(x.(n),d)
#     end
#     d[x] = y
# end

# function gpucopy_internal{A<:Array,T,N}(x::Par{A,T,N},d::ObjectIdDict)
#     haskey(d,x) && return d[x]
#     y = Par{CudaArray,T,N}()
#     for n in fieldnames(x)
#         isdefined(x,n) || continue
#         y.(n) = gpucopy_internal(x.(n),d)
#     end
#     d[x] = y
# end

# convert{A<:BaseArray}(::Type{A}, a::Par)=convert(A, a.arr)
# convert{A<:BaseArray}(::Type{Par}, a::A)=Par(a)

# TODO: both weights and training parameters are called param; setopt! vs params is confusing
# Probably should rename this back to Param after moving others to DynamicArray.
# How does caffe deal with this problem?

# Par(w; o...)=init(setopt!(Par{atype(w),eltype(w),ndims(w)}(); arr=w, o...))
# Par{A,T}(::Type{A}, ::Type{T}, d::Dims; o...)=Par(A(T,d); o...)
# Par{A,T}(::Type{A}, ::Type{T}, d1::Int, d::Int...; o...)=Par(A,T,tuple(d1,d...); o...)
# Par{T}(::Type{T}, d::Dims; o...)=Par((gpu()?CudaArray:Array),T,d; o...)
# Par{T}(::Type{T}, d::Int...; o...)=Par(T, d; o...)
# Par(d::Int...; o...)=Par(Float64, d; o...)

# TODO: create an rgen type: makes things serializable
# function init(p::Par, T::DataType=eltype(p), d::Dims=size(p.arr))
#     (size(p.arr)==d && eltype(p.arr)==T) || (p.arr = similar(p.arr, T, d))
#     # we want no init if params given in matrix
#     if !isempty(p.arr) && nz(p,:init,nothing)
#         initp = (!isdefined(p,:initp) ? () :
#                  isa(p.initp,Tuple) ? p.initp :
#                  (p.initp,))
#         p.init(p.arr, initp...)
#     end
#     return p
# end

# Base.isequal:
    # typeof(a) == typeof(b) || return false
    # size(a) == size(b) || return false

# DONE: back
# DONE: going back we should not zero the incremental dif!
