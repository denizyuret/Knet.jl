# TODO: update
# TODO: averaging

"""
`@knet function par(; dims, init, opts...)` creates a parameter array
of size `dims` which should be a tuple of Ints.  Some entries in dims
can be left as 0, in which case they will be inferred from the input.
`init` can be an Array, CudaArray, or a subtype of Rgen (please see
the Rgen doc).  Other `opts` include:

    * lr
    * l1reg
    * l2reg
    * adagrad
    * momentum
    * nesterov
    * average
"""
type Par <: Op; dims; init; initialized; out; dif; lr; l1reg; l2reg; adagrad; ada; momentum; mom; nesterov; nes; average; avg; 
    Par(;o...)=setopt!(new(); initialized=false, o...)
end

Kenv.kdef(:par,Par)

function setopt!(p::Par; o...)
    for (n,v) in o
        if in(n, fieldnames(p))
            p.(n)=v
        else
            # This happens regularly when binit gets passed to wdot etc.
            # Base.warn_once("setopt!: ignoring unrecognized option $n")
        end
    end
    p
end

function infersize(p::Par,ysize)
    psize = (isdefined(p,:init) && isa(p.init, BaseArray) ? size(p.init) :
             isdefined(p,:dims) ? p.dims : nothing)
    psize == nothing && return tuple(ysize)
    ysize == nothing && return tuple(psize)
    length(psize) == length(ysize) || throw(DimensionMismatch())
    dims = map(psize, ysize) do pi,yi
        pi==yi ? pi :
        pi==0  ? yi :
        yi==0  ? pi :
        throw(DimensionMismatch())
    end
    tuple(dims)
end

ninputs(::Par)=0
canoverwrite(::Par)=false
back_reads_x(::Par)=false
back_reads_y(::Par)=false

function back(p::Par, dy; y=nothing, o...)
    p.dif == nothing && (p.dif = dy)
    dy === p.dif || Base.warn_once("dy=$dy p.dif=$(p.dif)")
    y == nothing || y === p.out || Base.warn_once("y=$y p.out=$(p.out)")
end

function forw(p::Par, y; o...)
    if !p.initialized
        if !isdefined(p, :init)
            rgen!(Gaussian(0,0.01), y)
        elseif isa(p.init, BaseArray)
            copysync!(y, p.init)
        else
            rgen!(p.init, y)
        end
        p.out = y
        p.dims = size(y)
        p.dif = nothing
        p.initialized = true
    end
    @assert p.out === y "p.out=$(p.out) y=$y"
    return p.out
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
# initdiff(w::Par; fill=nothing, o...)=(similar!(w, :diff, w.arr); fillsync!=nothing && fillsync!(w.diff,fill); w)

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

# Use the dims option instead of:
# par(i::Integer, d::Integer...; o...)=par(; dims=(i,d...), o...)

# Use init=w for array initialization.
# par(w::AbstractArray, y; o...)=par(y; out0=w, dims=size(w), o...)

# """
# Rgen is an abstract type whose subtypes represent random distributions
# for parameter initialization.  Currently implemented subtypes are listed
# below.  They are used to specify initialization during Net construction.

#     * Gaussian(mean, std)
#     * Uniform(min, max)
#     * Constant(val)
#     * Identity(scale)
#     * Xavier()
# """
# abstract Rgen
# type Gaussian <: Rgen; mean; std; end
# type Uniform  <: Rgen; min; max; end
# type Constant <: Rgen; val; end
# type Identity <: Rgen; val; Identity(x=1)=new(x); end
# type Xavier <: Rgen; end

# par(y; o...)=(setopt!(Par(); initialized=false, o...), y)

