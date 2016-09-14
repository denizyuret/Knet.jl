# This is mostly copied from CUDArt to create an application specific
# memory manager.

# KnetPtr type holds a gpu (dev>=0) or cpu (dev=-1) allocated pointer.

type KnetPtr
    ptr::Ptr{Void}
    len::Int
    dev::Int
    parent
end

# We use the KnetPtrs type to keep track of allocated pointers: We
# need one per size per device.  KnetFree[i+2] will hold a dictionary
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

# The KnetPtr constructor tries to avoid actual allocation which is
# slow.  First it tries to find a previously allocated and garbage
# collected pointer in KnetFree[dev+2].  If not available it allocates
# a new one if we have more than 10^8 bytes of free memory.  Otherwise
# it tries running gc() and see if we get a pointer back.  Finally if
# all else fails, it calls knetgc which cleans up all allocated
# KnetPtrs on the current device and starts over.

function KnetPtr(nbytes::Integer)
    dev = gpu()
    ptrs = get!(KnetPtrs,KnetFree[dev+2],nbytes)
    if !isempty(ptrs.free)
        kp = KnetPtr(pop!(ptrs.free),nbytes,dev,nothing)
    elseif dev == -1 || gpufree() > 10^8
        ptr = knetMalloc(nbytes)
        kp = KnetPtr(ptr,nbytes,dev,nothing)
        ptrs.used += 1
    elseif (print("."); gc(); !isempty(ptrs.free))
        kp = KnetPtr(pop!(ptrs.free),nbytes,dev,nothing)
    else
        print((:knetgc,nbytes)); gpuinfo()
        knetgc()
        ptr = knetMalloc(nbytes)
        kp = KnetPtr(ptr,nbytes,dev,nothing)
        ptrs.used += 1
    end
    finalizer(kp, freeKnetPtr)
    return kp
end

# This is used to create a shared pointer.  We need to have the parent field to prevent premature gc.
function KnetPtr(parent::KnetPtr, offs::Integer, len::Integer)
    if len < 0 || offs < 1 || offs+len-1 > parent.len
        throw(BoundsError())
    end
    KnetPtr(parent.ptr+offs-1, len, parent.dev, parent)
end

# If you really want to clean up memory you need to call knetgc:
# Note that this only cleans the current device.

function knetgc()
    dev = gpu()
    gc_enable(false)
    for v in values(KnetFree[dev+2])
        if dev >= 0
            for p in v.free
                @cudart(:cudaFree,(Ptr{Void},),p)
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

meminfo()=[(k,v.used,length(v.free)) for (k,v) in KnetFree[gpu()+2]]


### KnetArray ###

if !isdefined(:KnetArray)
type KnetArray{T,N} # <: AbstractArray{T,N} # Never know when we'll inherit some inefficient fallback...
    ptr::KnetPtr
    dims::NTuple{N,Int}
end
end

KnetArray{T,N}(::Type{T}, dims::NTuple{N,Int})=KnetArray{T,N}(KnetPtr(sizeof(T)*prod(dims)), dims)
KnetArray(T::Type, dims::Int...)=KnetArray(T,dims)
KnetArray(T::Type, d::Integer...) = KnetArray(T,convert(Tuple{Vararg{Int}}, d))

typealias KnetMatrix{T} KnetArray{T,2}
typealias KnetVector{T} KnetArray{T,1}
typealias KnetVecOrMat{T} Union{KnetVector{T}, KnetMatrix{T}}


# KnetArray <- KnetArray
Base.convert{T,N}(::Type{KnetArray}, x::KnetArray{T,N}) = x
Base.convert{T,N}(::Type{KnetArray{T}}, x::KnetArray{T,N}) = x
Base.convert{T,N}(::Type{KnetArray{T,N}}, x::KnetArray{T,N}) = x
Base.convert{T,N,S}(::Type{KnetArray{T}}, x::KnetArray{S,N}) = convert(KnetArray{T,N}, x)
Base.convert{T,N,S}(::Type{KnetArray{T,N}}, x::KnetArray{S,N}) = convert(KnetArray{T,N},knetcopy!(Array(S, size(x)), 1, x, 1, length(x)))
Base.similar(a::KnetArray, T, dims::Dims) = KnetArray(T, dims)
Base.reshape{T}(a::KnetArray{T},dims::Dims)=(if dims==size(a); a; elseif prod(dims)!=length(a); throw(DimensionMismatch()); else; KnetArray{T,length(dims)}(a.ptr,dims); end)
Base.reshape(a::KnetArray, dims::Int...) = reshape(a, dims)
# KnetArray <- AbstractArray
Base.convert{T,N}(::Type{KnetArray}, x::AbstractArray{T,N}) = convert(KnetArray{T,N}, x)
Base.convert{T,N,S}(::Type{KnetArray{T}}, x::AbstractArray{S,N}) = convert(KnetArray{T,N}, x)
Base.convert{T,N,S}(::Type{KnetArray{T,N}}, x::AbstractArray{S,N}) = knetcopy!(KnetArray(T, size(x)), 1, convert(Array{T,N},x), 1, length(x))
# Array <- KnetArray
Base.convert{T,N}(::Type{Array}, x::KnetArray{T,N}) = convert(Array{T,N}, x)
Base.convert{T,N,S}(::Type{Array{T}}, x::KnetArray{S,N}) = convert(Array{T,N}, x)
Base.convert{T,N,S}(::Type{Array{T,N}}, x::KnetArray{S,N}) = convert(Array{T,N},knetcopy!(Array(S, size(x)), 1, x, 1, length(x)))
# Ptr <- KnetArray
Base.unsafe_convert{T}(::Type{Ptr{T}}, a::KnetArray) = Base.unsafe_convert(Ptr{T}, pointer(a))
Base.pointer{T}(a::KnetArray{T})=convert(Ptr{T}, a.ptr.ptr)
Base.pointer{T}(a::KnetArray{T},i)=convert(Ptr{T}, a.ptr.ptr + (i-1)*sizeof(T))


# AbstractArray interface
Base.size(a::KnetArray)=a.dims
Base.linearindexing(::KnetArray)=Base.LinearFast()

# We will implement indexing ranges as views not copies, if possible (when contiguous).
# For contiguous memory without stride all but the last >1 dimension must be full

# The original getindex(a,i:j...) for AbstractArray copies in abstractarray.jl:487,multidimensional.jl:184.
# function _getindex(l::LinearIndexing, A::AbstractArray, I::Union{Real, AbstractArray, Colon}...)

# which getindex ops does array implement?
# getindex(A::Array, i1::Real)
# getindex(A::Array, i1::Real, i2::Real, I::Real...)
# getindex(A::Array, I::UnitRange{Int})
# getindex(A::Array, c::Colon)
# getindex{T<:Real}(A::Array, I::Range{T})

# Julia #14770
# If I is shorter than ndims(A) but longer than 1 the remaining indices assumed =1
# Also extra 1's at the end of I are ignored

import Base: getindex, setindex!
using Base: to_indexes, index_shape, _getindex, _setindex!, linearindexing

# These two are not sufficient in spite of what the documentation says:
# display goes into an infinite loop!
# getindex{T}(A::KnetArray{T}, i::Int)=knetcopy!(T[0], 1, A, i, 1)[1]
# setindex!{T}(A::KnetArray{T}, v, i::Int)=knetcopy!(A, i, T[v], 1, 1)

# First deal with the easy cases: integer indices, a Colon or a UnitRange.

function getindex{T}(A::KnetArray{T}, I::Real...)
    J = to_indexes(I...)
    #TODO checkbounds(A,J...)
    i = sub2ind(size(A), J...)
    knetcopy!(T[0], 1, A, i, 1)[1]
end

function setindex!{T}(A::KnetArray{T}, v, I::Real...)
    J = to_indexes(I...)
    #TODO checkbounds(A,J...)
    i = sub2ind(size(A), J...)
    knetcopy!(A, i, T[v], 1, 1)
end

getindex(A::KnetArray, I::Colon)=reshape(A,length(A))

function setindex!{T}(A::KnetArray{T}, v, I::Colon)
    if isa(v,Number)
        knetfill!(A,T(v),1,length(A))
    elseif (isa(v,KnetArray) || isa(v,Array)) && (length(v)==length(A))
        eltype(v)==T || (v = convert(Array{T},v))
        knetcopy!(A,1,v,1,length(A))
    else
        @show (:setcolon,I)
        _setindex!(linearindexing(A), A, v, I)
    end
end

function getindex{T}(A::KnetArray{T}, I::UnitRange)
    #TODO checkbounds(A, I)
    off = 1+(first(I)-1)*sizeof(T)
    len = sizeof(T)*length(I)
    ptr = KnetPtr(A.ptr, off, len)
    KnetArray{T,1}(ptr, (length(I),))
end

function setindex!{T}(A::KnetArray{T}, v, I::UnitRange)
    if isa(v,Number)
        knetfill!(A,T(v),first(I),length(I))
    elseif (isa(v,KnetArray) || isa(v,Array)) && (length(v)==length(I))
        eltype(v)==T || (v = convert(Array{T},v))
        knetcopy!(A,first(I),v,1,length(I))
    else
        @show (:setunit,I)
        _setindex!(linearindexing(A), A, v, I)
    end
end

# TODO: the general getindex, setindex! work for 1 and 2 dimensions only, write general versions.
function indexparams{T,N}(A::KnetArray{T,N}, I::Union{Real, UnitRange, Colon}...)
    N > 2 && error("setindex for ndims > 2 not implemented yet")
    skipped = false
    nrows = nelts = 1
    subs1 = ones(Int,2)
    astep = length(A)
    for i=1:length(I)
        Ii = I[i]
        subs1[i] = first(Ii)
        Ai = size(A,i)
        if isa(Ii, Colon)
            Li = Ai
        elseif isa(Ii, Real)
            1 <= Ii <= Ai || throw(DimensionMismatch())
            Li = 1
        else
            1 <= first(Ii) <= last(Ii) <= Ai || throw(DimensionMismatch())
            Li = length(Ii)
        end
        nelts *= Li
        if !skipped
            nrows *= Li
            if Li < Ai
                skipped = true
                astep = stride(A,i+1)
            end
        end
    end
    ncols = div(nelts, nrows)
    firstindex = sub2ind(size(A),subs1...)
    return (nelts,nrows,ncols,firstindex,astep)
end

function setindex!{T,N}(A::KnetArray{T,N}, B, I::Union{Real, UnitRange, Colon}...)
    (nelts,nrows,ncols,firstindex,astep) = indexparams(A,I...)
    aptr0 = pointer(A, firstindex)
    if isa(B,Number)
        B = T(B)
        if ncols == 1
            if T <: Float32;    ccall((:fill_32,libknet8),Void,(Cint,Cfloat, Ptr{Cfloat}), nelts,B,aptr0)
            elseif T<: Float64; ccall((:fill_64,libknet8),Void,(Cint,Cdouble,Ptr{Cdouble}),nelts,B,aptr0)
            else error("$T not supported"); end
        else
            if T <: Float32;    ccall((:xfill_32,libknet8),Void,(Cint,Cint,Cfloat, Ptr{Cfloat}, Cint),nrows,ncols,B,aptr0,astep)
            elseif T<: Float64; ccall((:xfill_64,libknet8),Void,(Cint,Cint,Cdouble,Ptr{Cdouble},Cint),nrows,ncols,B,aptr0,astep)
            else error("$T not supported"); end
        end
    else
        length(B) == nelts || throw(DimensionMismatch())
        B = convert(KnetArray{T},B)
        if ncols == 1
            @cudart(:cudaMemcpyAsync,(Ptr{Void},Ptr{Void},Csize_t,UInt32,Ptr{Void}),
                    aptr0, B, nelts*sizeof(T), cudadir(A,B), C_NULL)
        else
            nrows *= sizeof(T); astep *= sizeof(T)
            ccall((:xcopy,libknet8),Void,(Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint), nrows, ncols, B, nrows, aptr0, astep)
        end
    end
    return A
end

function getindex{T,N}(A::KnetArray{T,N}, I::Union{Real, UnitRange, Colon}...)
    (nelts,nrows,ncols,firstindex,astep) = indexparams(A,I...)
    B1 = isa(I[1],Colon) ? size(A,1) : length(I[1])
    B2 = isa(I[2],Colon) ? size(A,2) : length(I[2])
    if ncols == 1
        off = 1+(firstindex-1)*sizeof(T)
        len = nrows*sizeof(T)
        ptr = KnetPtr(A.ptr, off, len)
        KnetArray{T,2}(ptr, (B1,B2))
    else
        B = similar(A, (B1,B2))
        nrows *= sizeof(T); astep *= sizeof(T)
        ccall((:xcopy,libknet8),Void,(Cint,Cint,Ptr{Void},Cint,Ptr{Void},Cint),
              nrows, ncols, pointer(A,firstindex), astep, B, nrows)
        return B
    end
end

# If I is contiguous return a shared pointer, otherwise call Julia getindex.
function getindex1{T}(A::KnetArray{T}, I::Union{Real, UnitRange, Colon}...)
    asize = size(A)
    bsize = zeros(Int,length(I))
    bstart = 1
    for i in 1:length(asize)
        if i <= length(bsize)
            if I[i]==(:); bi=size(A,i); else; bi=length(I[i]); end
            bsize[i] = bi
        else
            bi = 1
        end
        if bi < asize[i]
            if prod(asize[i+1:end]) > 1
                return _getindex(linearindexing(A), A, I...)
            else
                bstart = 1+(first(I[i])-1)*stride(A,i)
                break
            end
        end
    end
    off = 1+(bstart-1)*sizeof(T)
    len = prod(bsize)*sizeof(T)
    ptr = KnetPtr(A.ptr, off, len)
    KnetArray{T,length(bsize)}(ptr, bsize)
end

# hcat(v,v): I = (Colon(),1:1) I = (Colon(),2:2)
# vcat(v,v): uses single index
# hcat(m,m): I = (Colon(),1:5) I = (Colon(),6:10)
# vcat(m,m): I = (1:3,Colon()) I = (4:6,Colon())


# see if we miss anything:
# function setindex!{T}(A::KnetArray{T}, v, I...)
#     @show (:setindex,I)
#     _setindex!(linearindexing(A),A,v,I...)
# end


# Benchmarks in μs for hcat and vcat: a=rand(1000,1000) v=rand(1000), t=v'
#
#		cpu	gpu	g->c->g	vkernel
# hcat(a,a)	2350	225	16160
# hcat(a,v)	1230	115	6490
# hcat(v,a)	1220	120	6490
# hcat(v,v)	3.53	12.53	48.49
# vcat(a,a)	2630	10980	16590	665
# vcat(a,t)	1350	10860	6550	338
# vcat(t,a)	1360	10850	6570	338
# vcat(v,v)	2.13	12.33	45.40	13.58

function hcat{T}(a::KnetVecOrMat{T}, b::KnetVecOrMat{T})
    size(a,1)==size(b,1) || throw(DimensionMismatch())
    c1 = size(a,1)
    c2 = size(a,2) + size(b,2)
    c = KnetArray(T, (c1,c2))
    c[:,1:size(a,2)] = a
    c[:,1+size(a,2):end] = b
    return c
end

function vcat{T}(a::KnetVector{T}, b::KnetVector{T})
    c = KnetArray(T, length(a)+length(b))
    c[1:length(a)] = a
    c[1+length(a):end] = b
    return c
end

function vcat{T}(a::KnetVecOrMat{T}, b::KnetVecOrMat{T})
    size(a,2)==size(b,2) || throw(DimensionMismatch())
    c1 = size(a,1) + size(b,1)
    c2 = size(a,2)
    c = KnetArray(T, (c1,c2))
    c[1:size(a,1),:] = a
    c[1+size(a,1):end,:] = b
    return c
end

function cat{T}(d, a::KnetVecOrMat{T}, b::KnetVecOrMat{T})
    if     d==1; vcat(a,b)
    elseif d==2; hcat(a,b)
    else error("cat($d) not implemented.")
    end
end

# Generalizing low level copy using linear indexing to/from gpu arrays:
# copy!{T}(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer)

typealias Kcopy{T} Union{Array{T},KnetArray{T}}

function knetcopy!{T}(dest::Kcopy{T}, doffs::Integer, src::Kcopy{T}, soffs::Integer, n::Integer; stream=C_NULL)
    n == 0 && return dest
    isbits(T) || error("knetcopy! only works for isbits arrays.")
    if n < 0 || soffs < 1 || doffs < 1 || soffs+n-1 > length(src) || doffs+n-1 > length(dest)
        throw(BoundsError())
    end
    if (isa(src,KnetArray) && src.ptr.dev >= 0) || (isa(dest,KnetArray) && dest.ptr.dev >= 0)
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

for S in (32,64); T = Symbol("Float$S"); F = "fill_$S"
    @eval function knetfill!(a::KnetArray{$T},v::$T,off,len)
        ccall(($F,$libknet8),Void,(Cint,$T,Ptr{$T}),len,v,pointer(a,off))
    end
end

Base.fill!{T}(a::KnetArray{T},x)=(knetfill!(a,T(x),1,length(a));a)
Base.zeros{T}(a::KnetArray{T})=fill!(similar(a),zero(T))
Base.ones{T}(a::KnetArray{T})=fill!(similar(a),one(T))

# To be able to load/save KnetArrays:
if isdir(Pkg.dir("JLD"))
    import JLD: writeas, readas
    type KnetJLD; a::Array; end
    writeas(c::KnetArray) = KnetJLD(Array(c))
    readas(d::KnetJLD) = KnetArray(d.a)
end

# These are defined for AbstractArrays, so we can remove them eventually:
Base.size{T,N}(x::KnetArray{T,N},i::Integer)=(if i>N; 1; else; size(x)[i]; end)
Base.stride(x::KnetArray,i::Integer)=(if i>ndims(x); length(x); else; s=1; for n=1:(i-1); s*=size(x,n); end; s; end)
Base.summary(a::KnetArray) = string(Base.dims2string(size(a)), " ", typeof(a))
Base.ndims{T,N}(a::KnetArray{T,N})=N
Base.eltype{T}(x::KnetArray{T})=T
Base.elsize{T}(::KnetArray{T}) = sizeof(T)
Base.length(a::KnetArray)=prod(size(a))
Base.endof(a::KnetArray) = length(a)
Base.first(a::KnetArray) = a[1]
Base.eachindex(a::KnetArray) = (1:length(a))
Base.similar{T}(a::KnetArray{T})               = similar(a, T, size(a))
Base.similar(   a::KnetArray, T)               = similar(a, T, size(a))
Base.similar{T}(a::KnetArray{T}, dims::Dims)   = similar(a, T, dims)
Base.similar{T}(a::KnetArray{T}, dims::Int...) = similar(a, T, dims)
Base.similar(   a::KnetArray, T, dims::Int...) = similar(a, T, dims)

import AutoGrad: zeroslike, sum_outgrads, OneHot
zeroslike(a::KnetArray)=zeros(a)
sum_outgrads{T}(a::KnetArray{T},b::KnetArray{T})=(a+b)
sum_outgrads(a::KnetArray,b::OneHot)=setindex!(a,sum_outgrads(getindex(a,b.index...),b.value),b.index...)

# For printing without copying the whole KnetArray and without inheriting AbstractArray:
type KnetDisplay{T,N} <: AbstractArray{T,N}; a::KnetArray{T,N}; end
getindex(a::KnetDisplay, i...) = getindex(a.a, i...)
size(a::KnetDisplay) = size(a.a)
summary(a::KnetDisplay) = string(Base.dims2string(size(a)), " ", typeof(a.a))
display(a::KnetArray) = display(KnetDisplay(a))
