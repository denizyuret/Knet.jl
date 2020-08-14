using Knet.KnetArrays: KnetPtr, KnetArray
using Knet.Ops20: RNN

const JLDMODE=Val(0)
const GPUMODE=Val(1)
const CPUMODE=Val(2)

# Do not use type asserts because type may change
serialize(x) = _ser(x,IdDict(),JLDMODE)
gpucopy(x)   = _ser(x,IdDict(),GPUMODE)
cpucopy(x)   = _ser(x,IdDict(),CPUMODE)

function _ser(x::KnetPtr,s::IdDict,::typeof(JLDMODE))
    if !haskey(s,x)
        if isa(x.ptr, Cptr) && (x.dev >= 0)
            a = Array{UInt8}(undef,x.len)
            # @cudart(cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),pointer(a),x.ptr,x.len,2)
            unsafe_copyto!(pointer(a), CuPtr{UInt8}(UInt(x.ptr)), x.len)
            s[x] = KnetPtr(a,x.len,-1,nothing)
        elseif isa(x.ptr, Array{UInt8,1})
            if gpu() >= 0
                s[x] = KnetPtr(x.len)
                # @cudart(cudaMemcpy,(Cptr,Cptr,Csize_t,UInt32),s[x].ptr,pointer(x.ptr),x.len,1)
                unsafe_copyto!(CuPtr{UInt8}(UInt(s[x].ptr)), pointer(x.ptr), x.len)
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
            s[x] = copy(reshape(reinterpret(eltype(x),view(x.ptr.ptr,1:sizeof(T)*length(x))),size(x)))
        else
            s[x] = KnetArray{T,N}(_ser(x.ptr,s,m),x.dims)
        end
    end
    return s[x]
end

function _ser(x::RNN, s::IdDict, m::Val)
    if !haskey(s,x)
        # we need rd,dd only if there is a gpu, we are not in cpumode,
        # and if we are in jldmode we are loading, not saving
        # if (gpu() >= 0 && m != CPUMODE && !(m == JLDMODE && x.rnnDesc != nothing))
        #     dd = DD(dropout=x.dropout,seed=x.seed)
        #     rd = RD(x.hiddenSize,x.numLayers,dd,x.inputMode,x.direction,x.mode,x.algo,x.dataType)
        # else
        #     rd = dd = nothing
        # end

        # 20200806: We no longer need to load/save rd/dd: rnnforw will construct as needed.
        rd = dd = nothing

        # dx, dhx, dcx are temporary fields used by rnnback, they do not need to be copied
        # gcnode sets dx.ptr to C_NULL which breaks serialize, best not to try
        s[x] = RNN(_ser(x.w,s,m), _ser(x.h,s,m), _ser(x.c,s,m), x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.seed, x.inputMode, x.direction, x.mode, x.algo, x.dataType, rd, dd, nothing, nothing, nothing)
    end
    return s[x]
end

# Partially fixes the issue: when KA converts to A because no gpu, surrounding parametric types remain Param{KA}.
# However other container types that include KnetArray may still have an inconsistent parametric type problem.
_ser(x::Param, s::IdDict, m::Val)=(haskey(s,x) ? s[x] : s[x]=Param(_ser(x.value,s,m),_ser(x.opt,s,m)))

_ser(x::KnetArray,s::IdDict,::typeof(GPUMODE))=x
_ser(x::KnetArray,s::IdDict,::typeof(CPUMODE))=(haskey(s,x) ? s[x] : s[x]=Array(x))
_ser(x::Array, s::IdDict, m::Val) = (haskey(s, x) ? s[x] : s[x] = _ser_array_t(x, eltype(x), s, m))

function _ser_array_t(@nospecialize(x), T, s::IdDict, m::Val) 
    if !isbitstype(T)
        map(xi->_ser(xi,s,m), x)
    elseif m === GPUMODE
        KnetArray(x)
    else
        x  # Note: deepcopy actually copies here, we don't
    end
end


# Generic serialization rules from deepcopy.jl in julia/base:
_ser(x::Union{Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,UnionAll,Task,Regex},::IdDict,::Val) = x
_ser(x::Tuple, s::IdDict, m::Val) = ntuple(i->_ser(x[i], s, m), length(x))
_ser(x::Module, ::IdDict, ::Val) = error("serialize of Modules not supported")
_ser(x::Core.SimpleVector, s::IdDict,m::Val) = (haskey(s, x) ? s[x] : s[x] = Core.svec(Any[_ser(x[i], s, m) for i = 1:length(x)]...))
_ser(x::String, s::IdDict,::Val) = (haskey(s, x) ? s[x] : s[x] = (GC.@preserve x unsafe_string(pointer(x), sizeof(x))))

function _ser(@nospecialize(x), stackdict::IdDict, m::Val)
    T = typeof(x)::DataType
    nf = nfields(x)
    if T.mutable
        if haskey(stackdict, x)
            return stackdict[x]
        end
        y = ccall(:jl_new_struct_uninit, Any, (Any,), T)
        stackdict[x] = y
        for i in 1:nf
            if isdefined(x, i)
                xi = getfield(x, i)
                xi = _ser(xi, stackdict, m)  # removed ::typeof(xi)
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i-1, xi)
            end
        end
    elseif nf == 0 || isbitstype(T)
        y = x
    else
        flds = Vector{Any}(undef, nf)
        for i in 1:nf
            if isdefined(x, i)
                xi = getfield(x, i)
                xi = _ser(xi, stackdict, m)  # removed ::typeof(xi)
                flds[i] = xi
            else
                nf = i - 1 # rest of tail must be undefined values
                break
            end
        end
        y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, flds, nf)
    end
    return y
end

function _ser(x::Union{Dict,IdDict}, s::IdDict, m::Val)
    if haskey(s, x)
        return s[x] # removed ::typeof(x)
    end
    if isbitstype(eltype(x))
        return (s[x] = x)
    end
    dest = empty(x)
    s[x] = dest
    for (k, v) in x
        dest[_ser(k, s, m)] = _ser(v, s, m)
    end
    dest
end
