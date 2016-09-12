# This is mostly copied from CUDArt to create an application specific
# memory manager.

# KnetPtr type holds a gpu allocated pointer.

type KnetPtr
    ptr::Ptr{Void}
    len::Int
    dev::Int
end

# We use the KnetPtrs type to keep track of allocated pointers: We
# need one per device per size.  KnetFree[i+2] will hold a dictionary
# from sizes to KnetPtrs for device i.

type KnetPtrs
    used::Int
    free::Array{Ptr{Void},1}
    KnetPtrs()=new(0,Array(Ptr{Void},0))
end

# When Julia gc reclaims a KnetPtr object, the finalizer does not
# actually release the memory, but inserts it in the KnetFree dict
# keyed by length in bytes so it can be reused.

if !isdefined(:KnetFree)
    KnetFree = [ Dict{Int,KnetPtrs}() for i=1:gpucount()+1 ]
end

function freeKnetPtr(p::KnetPtr)
    ptrs = KnetFree[p.dev+2][p.len]
    push!(ptrs.free,p.ptr)
end

function KnetPtr(nbytes::Integer)
    dev = gpu()
    ptrs = get!(KnetPtrs,KnetFree[dev+2],nbytes)
    if !isempty(ptrs.free)
        kp = KnetPtr(pop!(ptrs.free),nbytes,dev)
    elseif gpufree() > 10^8
        ptr = knetMalloc(nbytes)
        kp = KnetPtr(ptr,nbytes,dev)
        ptrs.used += 1
    elseif (print("."); gc(); !isempty(ptrs.free))
        kp = KnetPtr(pop!(ptrs.free),nbytes,dev)
    else
        print((:knetgc,nbytes)); gpuinfo()
        knetgc()
        ptr = knetMalloc(nbytes)
        kp = KnetPtr(ptr,nbytes,dev)
        ptrs.used += 1
    end
    finalizer(kp, freeKnetPtr)
    return kp
end

# If you really want to clean up memory you need to call knetgc:
# Note that this only cleans the current device.

function knetgc()
    dev = gpu()
    gc_enable(false)
    for v in values(KnetFree[dev+2])
        if dev >= 0
            for p in v.free
                knetFree(p)
            end
        end
        v.used -= length(v.free)
        empty!(v.free)
    end
    gc_enable(true)
end

function knetMalloc(nbytes::Int)
    if gpu() >= 0
        ptr = Ptr{Void}[C_NULL]
        @cudart(:cudaMalloc,(Ptr{Ptr{Void}},Csize_t),ptr,nbytes)
        return ptr[1]
    else
        convert(Ptr{Void}, pointer(Array(UInt8,nbytes)))
    end
end

function knetFree(p::Ptr{Void})
    # TODO: Do we need the active device be p's device?
    @cudart(:cudaFree,(Ptr{Void},),p)
end

if !isdefined(:KnetArray)
type KnetArray{T,N} # <: AbstractArray{T,N} # TODO: uncomment and deal with ambiguities
    ptr::KnetPtr
    dims::NTuple{N,Int}
end
end

KnetArray{T,N}(::Type{T}, dims::NTuple{N,Int})=KnetArray{T,N}(KnetPtr(sizeof(T)*prod(dims)), dims)
KnetArray(T::Type, dims::Int...)=KnetArray(T,dims)
KnetArray(T::Type, d::Integer...) = KnetArray(T,convert(Tuple{Vararg{Int}}, d))

typealias KnetMatrix{T} KnetArray{T,2}
typealias KnetVector{T} KnetArray{T,1}

Base.convert{T,N}(::Type{KnetArray}, x::KnetArray{T,N}) = x
Base.convert{T,N}(::Type{KnetArray{T}}, x::KnetArray{T,N}) = x
Base.convert{T,N}(::Type{KnetArray{T,N}}, x::KnetArray{T,N}) = x
Base.convert{T,N}(::Type{KnetArray}, x::AbstractArray{T,N}) = convert(KnetArray{T,N}, x)
Base.convert{T,N,S}(::Type{KnetArray{T}}, x::AbstractArray{S,N}) = convert(KnetArray{T,N}, x)
Base.convert{T,N,S}(::Type{KnetArray{T,N}}, x::AbstractArray{S,N}) = knetcopy!(KnetArray(T, size(x)), 1, convert(Array{T,N},x), 1, length(x))
Base.convert{T,N}(::Type{Array}, x::KnetArray{T,N}) = convert(Array{T,N}, x)
Base.convert{T,N,S}(::Type{Array{T}}, x::KnetArray{S,N}) = convert(Array{T,N}, x)
Base.convert{T,N,S}(::Type{Array{T,N}}, x::KnetArray{S,N}) = convert(Array{T,N},knetcopy!(Array(S, size(x)), 1, x, 1, length(x)))
Base.convert{T,N,S}(::Type{KnetArray{T}}, x::KnetArray{S,N}) = convert(KnetArray{T,N}, x)
Base.convert{T,N,S}(::Type{KnetArray{T,N}}, x::KnetArray{S,N}) = convert(KnetArray{T,N},knetcopy!(Array(S, size(x)), 1, x, 1, length(x)))

Base.similar{T}(a::KnetArray{T})=KnetArray(T,size(a))
Base.similar{T}(a::KnetArray{T},dims::Dims)=KnetArray(T,dims)
Base.similar{T}(a::KnetArray{T},dims::Int...)=KnetArray(T,dims)
Base.unsafe_convert{T}(::Type{Ptr{T}}, a::KnetArray) = Base.unsafe_convert(Ptr{T}, pointer(a))
Base.pointer{T}(a::KnetArray{T})=convert(Ptr{T}, a.ptr.ptr)
Base.pointer{T}(a::KnetArray{T},i)=convert(Ptr{T}, a.ptr.ptr + (i-1)*sizeof(T))
Base.reshape{T}(a::KnetArray{T},dims::Dims)=(prod(dims)==length(a)||throw(DimensionMismatch()); KnetArray{T,length(dims)}(a.ptr,dims))

# AbstractArray interface
Base.size(a::KnetArray)=a.dims
Base.linearindexing(::KnetArray)=Base.linearfast()
Base.getindex{T}(a::KnetArray{T}, i::Integer)=knetcopy!(T[0], 1, a, i, 1)[1]
Base.setindex!{T}(a::KnetArray{T}, v, i::Integer)=knetcopy!(a, i, T[convert(T,v)], 1, 1)

# These are defined for AbstractArrays, so we can remove it eventually:
Base.length(a::KnetArray)=prod(size(a))
Base.ndims(a::KnetArray)=length(size(a))
Base.size(x::KnetArray,i::Integer)=(if i>ndims(x); 1; else; size(x)[i]; end)
Base.eltype{T}(x::KnetArray{T})=T
Base.stride(x::KnetArray,i::Integer)=(if i>ndims(x); length(x); else; s=1; for n=1:(i-1); s*=size(x,n); end; s; end)
Base.summary(a::KnetArray) = string(Base.dims2string(size(a)), " ", typeof(a))
Base.eachindex(a::KnetArray) = (1:length(a))
import AutoGrad: sum_outgrads
sum_outgrads{T}(a::KnetArray{T},b::KnetArray{T})=(a+b)

# Generalizing low level copy using linear indexing to/from gpu arrays:
# copy!{T}(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer)

typealias Kcopy{T} Union{Array{T},SubArray{T},KnetArray{T}}

function knetcopy!{T}(dest::Kcopy{T}, doffs::Integer, src::Kcopy{T}, soffs::Integer, n::Integer; stream=C_NULL)
    n == 0 && return dest
    isbits(T) || error("knetcopy! only works for isbits arrays.")
    if n < 0 || soffs < 1 || doffs < 1 || soffs+n-1 > length(src) || doffs+n-1 > length(dest)
        throw(BoundsError())
    end
    if (isa(src,KnetArray) && src.ptr.dev >= 0) || (isa(dest,KnetArray) && dest.ptr.dev >= 0)
        # TODO: does this copy between devices?
        @cudart(:cudaMemcpyAsync,(Ptr{Void},Ptr{Void},Csize_t,UInt32,Ptr{Void}),
                pointer(dest,doffs), pointer(src,soffs), n*sizeof(T), cudadir(dest, src), stream)
    else
        Base.warn_once("Using KnetArray on the CPU.")
        Base.unsafe_copy!(pointer(dest,doffs), pointer(src,soffs), n)
        # error("GPU is inactive, please use gpu(true) or gpu(n) to use KnetArray.")
    end
    return dest
end

function cudadir(a,b)
    deva = isa(a,KnetArray) && a.ptr.dev >= 0
    devb = isa(b,KnetArray) && b.ptr.dev >= 0
    if !deva && !devb; return 0
    elseif deva && !devb; return 1
    elseif !deva && devb; return 2
    elseif deva && devb;  return 3
    end
end

# TODO: this will be fixed when we inherint from AbstractArray.
Base.display(x::KnetArray)=(print("KnetArray ");display(convert(Array,x)))

meminfo()=[(k,v.used,length(v.free)) for (k,v) in KnetFree[gpu()+2]]

# To be able to load/save KnetArrays:
if isdir(Pkg.dir("JLD"))
    import JLD: writeas, readas
    type _KnetArray; a::Array; end
    writeas(c::KnetArray) = _KnetArray(Array(c))
    readas(d::_KnetArray) = KnetArray(d.a)
end
