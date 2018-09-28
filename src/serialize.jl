struct RnnJLD; inputSize; hiddenSize; numLayers; dropout; inputMode; direction; mode; algo; dataType; w; end
struct KnetJLD; a::Array ; end
struct ParamJLD; value; opt; ParamJLD(x) = new(x); end

const JLDMODE=Val(0)
const GPUMODE=Val(1)
const CPUMODE=Val(2)

serialize(x) = serialize_internal(x,IdDict(),JLDMODE)
gpu(x)       = serialize_internal(x,IdDict(),GPUMODE)
cpu(x)       = serialize_internal(x,IdDict(),CPUMODE)

serialize_internal(x::KnetArray,stackdict::IdDict,::typeof(JLDMODE)) = KnetJLD(Array(x))
serialize_internal(x::KnetArray,stackdict::IdDict,::typeof(GPUMODE))=x
serialize_internal(x::KnetArray,stackdict::IdDict,::typeof(CPUMODE))=Array(x)
serialize_internal(x::Array,stackdict::IdDict,::typeof(GPUMODE))=KnetArray(x)
serialize_internal(x::Array,stackdict::IdDict,::typeof(CPUMODE))=x

serialize_internal(d::KnetJLD,stackdict::IdDict,::typeof(JLDMODE))  =(gpu() >= 0 ? KnetArray(d.a) : d.a)

serialize_internal(x::Param{<:KnetArray},stackdict::IdDict,::typeof(JLDMODE))=isdefined(x,:opt) ? ParamJLD(serialize_internal(x.value,stackdict,JLDMODE),serialize_internal(x.opt,stackdict,JLDMODE)) : ParamJLD(serialize_internal(x.value,stackdict,JLDMODE))        

serialize_internal(x::Param{<:KnetArray},stackdict::IdDict,::typeof(CPUMODE))=isdefined(x,:opt) ? Param(serialize_internal(x.value,stackd\
ict,JLDMODE),serialize_internal(x.opt,stackdict,JLDMODE)) : Param(serialize_internal(x.value,stackdict,CPUMODE))

erialize_internal(x::Param{<:Array},stackdict::IdDict,::typeof(GPUMODE))=isdefined(x,:opt) ? Param(serialize_internal(x.value,stackd\
ict,GPUMODE),serialize_internal(x.opt,stackdict,GPUMODE)) : Param(serialize_internal(x.value,stackdict,GPUMODE))

serialize_internal(x::ParamJLD,stackdict::IdDict,::typeof(JLDMODE))=isdefined(x,:opt) ? Param(serialize_internal(x.value,stackdict,JLDMODE),serialize_internal(x.value,stackdict,JLDMODE)) : Param(serialize_internal(x.value,stackdict,JLDMODE))

serialize_internal(x::RNN,stackdict::IdDict,::typeof(JLDMODE)) = RnnJLD(x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.inputMode, x.direction, x.mode, x.algo, x.dataType, serialize_internal(x.w,stackdict,JLDMODE))        

serialize_internal(x::RNN,stackdict::IdDict,mode::Val) = (x.w = serialize_internal(x.w,stackdict,mode); return x)

serialize_internal(r::RnnJLD,stackdict::IdDict,::typeof(JLDMODE))         = ((x,w) = rnninit(r.inputSize, r.hiddenSize, numLayers=r.numLayers, dropout=r.dropout,
                                                                           skipInput=(r.inputMode==1), bidirectional=(r.direction==1),
                                                                           rnnType=(:relu,:tanh,:lstm,:gru)[1+r.mode], algo=r.algo, dataType=r.dataType); x.w = serialize_internal(r.w,stackdict,JLDMODE); x)

serialize_internal(x::Union{Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,Task},
                  stackdict::IdDict,::Val) = x
serialize_internal(x::Tuple, stackdict::IdDict,mode::Val) =
    ntuple(i->serialize_internal(x[i], stackdict, mode), length(x))
serialize_internal(x::Module, stackdict::IdDict,::Val) = error("serialize of Modules not supported")

function serialize_internal(x::Core.SimpleVector, stackdict::IdDict,mode::Val)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    y = Core.svec(Any[serialize_internal(x[i], stackdict, mode) for i = 1:length(x)]...)
    stackdict[x] = y
    return y
end

function serialize_internal(x::String, stackdict::IdDict,::Val)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    y = x
    stackdict[x] = y
    return y
end

function serialize_internal(@nospecialize(x), stackdict::IdDict,mode::Val)
    T = typeof(x)::DataType
    nf = nfields(x)
    (isbitstype(T) || nf == 0) && return x
    if haskey(stackdict, x)
        return stackdict[x]
    end
    y = ccall(:jl_new_struct_uninit, Any, (Any,), T)
    if T.mutable
        stackdict[x] = y
    end
    for i in 1:nf
        if isdefined(x,i)
            ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i-1,
                  serialize_internal(getfield(x,i), stackdict,mode))
        end
    end
    return y::T
end

function serialize_internal(x::Array, stackdict::IdDict,mode::Val)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    _serialize_array_t(x, eltype(x), stackdict,mode)
end

function _serialize_array_t(@nospecialize(x), T, stackdict::IdDict,mode::Val)
    if isbitstype(T)
        return (stackdict[x]=x)
    end
    y = map(xi->serialize_internal(xi,stackdict,mode), x)
    stackdict[x] = y
    return y
end

function serialize_internal(x::Union{Dict,IdDict}, stackdict::IdDict,mode::Val)
    if haskey(stackdict, x)
        return stackdict[x]
    end

    if isbitstype(eltype(x))
        return (stackdict[x] = x)
    end

    dest = typeof(x) <: Dict ? Dict() : IdDict()
    stackdict[x] = dest
    for (k, v) in x
        dest[serialize_internal(k, stackdict,mode)] = serialize_internal(v, stackdict,mode)
    end
    dest
end
