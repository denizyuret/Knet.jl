const JLDMODE=Val(0)
const GPUMODE=Val(1)
const CPUMODE=Val(2)

serialize(x) = _ser(x,IdDict(),JLDMODE)

"Return a copy of `x` with all its arrays transferred to GPU."
gpucopy(x)   = _ser(x,IdDict(),GPUMODE)

"Return a copy of `x` with all its arrays transferred to CPU."
cpucopy(x)   = _ser(x,IdDict(),CPUMODE)

function _ser(x::KnetPtr,s::IdDict,::typeof(JLDMODE))
    if !haskey(s,x)
        if isa(x.ptr, Cptr) && (x.dev >= 0)
            a = Array{UInt8}(undef,x.len)
            @cudart(cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),pointer(a),x.ptr,x.len,2)
            s[x] = KnetPtr(a,x.len)
        elseif isa(x.ptr, Array{UInt8,1})
            if gpu() >= 0
                g = knetMalloc(x.len); if g === nothing; error("No space left on GPU."); end
                @cudart(cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),g,pointer(x.ptr),x.len,1)
                s[x] = KnetPtr(g,x.len,gpu())
            else
                s[x] = x  # Leave conversion to array to KnetArray
            end
        else
            error("Unrecognized KnetPtr")
        end
    end
    return s[x]
end

function _ser(x::KnetArray{T,N},s::IdDict,m::typeof(JLDMODE)) where {T,N}
    if !haskey(s,x)
        if isa(x.ptr.ptr, Array) && gpu() < 0
            s[x] = reshape(reinterpret(eltype(x),view(x.ptr.ptr,1:sizeof(T)*length(x))),size(x))
        else
            s[x] = KnetArray{T,N}(_ser(x.ptr,s,m),x.dims)
        end
    end
    return s[x]
end

function _ser(x::RNN, s::IdDict, m::typeof(JLDMODE))
    if !haskey(s,x)
        dropoutDesc = (x.dropoutDesc != nothing || gpu() < 0 ? nothing : DD(handle=gethandle(),dropout=x.dropout,seed=x.seed))
        rnnDesc = (x.rnnDesc != nothing || gpu() < 0 ? nothing : RD(x.hiddenSize,x.numLayers,dropoutDesc,x.inputMode,x.direction,x.mode,x.algo,x.dataType))
        s[x] = RNN(_ser(x.w,s,m), _ser(x.h,s,m), _ser(x.c,s,m), x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.seed, x.inputMode, x.direction, x.mode, x.algo, x.dataType, rnnDesc, dropoutDesc, _ser(x.dx,s,m), _ser(x.dhx,s,m), _ser(x.dcx,s,m))
    end
    return s[x]
end

# Partially fixes the issue: when KA converts to A because no gpu, surrounding parametric types remain Param{KA}.
# However other container types that include KnetArray may still have an inconsistent parametric type problem.
_ser(x::Param, s::IdDict, m::Val)=(haskey(s,x) ? s[x] : s[x]=Param(_ser(x.value,s,m),_ser(x.opt,s,m)))

_ser(x::KnetArray,s::IdDict,::typeof(GPUMODE))=x
_ser(x::KnetArray,s::IdDict,::typeof(CPUMODE))=(haskey(s,x) ? s[x] : s[x]=Array(x))
_ser(x::Array,s::IdDict,::typeof(GPUMODE))=(haskey(s,x) ? s[x] : s[x]=KnetArray(x))
_ser(x::Array,s::IdDict,::typeof(CPUMODE))=x


# Generic serialization rules from deepcopy.jl
_ser(x::Union{Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,Task},::IdDict,::Val) = x
_ser(x::Tuple, s::IdDict, m::Val) = ntuple(i->_ser(x[i], s, m), length(x))
_ser(x::Module, ::IdDict, ::Val) = error("serialize of Modules not supported")
_ser(x::Core.SimpleVector, s::IdDict,m::Val) = (haskey(s, x) ? s[x] : s[x] = Core.svec(Any[_ser(x[i], s, m) for i = 1:length(x)]...))
_ser(x::String, s::IdDict,::Val) = (haskey(s, x) ? s[x] : s[x] = (GC.@preserve x unsafe_string(pointer(x), sizeof(x))))
_ser(x::Array, s::IdDict, m::Val) = (haskey(s, x) ? s[x] : s[x] = _ser_array_t(x, eltype(x), s, m))
_ser_array_t(@nospecialize(x), T, s::IdDict, m::Val) = (isbitstype(T) ? x : map(xi->_ser(xi,s,m), x))

function _ser(@nospecialize(x), s::IdDict, m::Val)
    T = typeof(x)::DataType
    nf = nfields(x)
    (isbitstype(T) || nf == 0) && return x
    if haskey(s, x)
        return s[x]
    end
    y = ccall(:jl_new_struct_uninit, Any, (Any,), T)
    if T.mutable
        s[x] = y
    end
    for i in 1:nf
        if isdefined(x,i)
            ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i-1,
                  _ser(getfield(x,i), s, m))
        end
    end
    return y::T
end

function _ser(x::Union{Dict,IdDict}, s::IdDict,m::Val)
    if haskey(s, x)
        return s[x]
    end
    if isbitstype(eltype(x))
        return (s[x] = x)
    end
    dest = typeof(x) <: Dict ? Dict() : IdDict()
    s[x] = dest
    for (k, v) in x
        dest[_ser(k, s, m)] = _ser(v, s, m)
    end
    dest
end
