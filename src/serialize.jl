struct RnnJLD; inputSize; hiddenSize; numLayers; dropout; inputMode; direction; mode; algo; dataType; w; end
struct KnetJLD; a::Array ; end
struct ParamJLD; value; end

mode = true
serialize(x) = serialize_internal(x, IdDict{Any,Any}(mode=>0))
gpu(x)       = serialize_internal(x, IdDict{Any,Any}(mode=>1))
cpu(x)       = serialize_internal(x, IdDict{Any,Any}(mode=>2))


@inline function serialize_internal(x::KnetArray,stackdict::IdDict)   
    if stackdict[mode]==0
        KnetJLD(Array(x))
    elseif stackdict[mode]==1
        x
    else
        Array(x)
    end
end

serialize_internal(d::KnetJLD,stackdict::IdDict)        = (gpu() >= 0 ? KnetArray(d.a) : d.a)

@inline function serialize_internal(x::Param,stackdict::IdDict)          
    if stackdict[mode]==0
        ParamJLD(Array(x))
    elseif stackdict[mode]==1
        if typeof(x.value) <:Array
            isdefined(x,:opt) ? Param(KnetArray(x.value),KnetArray(x.opt)) : Param(KnetArray(x.value))
        else
            x
        end
    else
        if typeof(x.value) <: KnetArray
            isdefined(x,:opt) ? Param(Array(x.value),Array(x.opt)) : Param(Array(x.value))
        else
            x
        end
    end
end
serialize_internal(x::ParamJLD,stackdict::IdDict) = Param(serialize_internal(x.value,stackdict))

function serialize_internal(x::RNN,stackdict::IdDict)       
    if stackdict[mode]==0
          RnnJLD(x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.inputMode, x.direction, x.mode, x.algo, x.dataType, serialize_internal(x.w,stackdict))
    else
        x.w = serialize_internal(x.w,stackdict)
        return x
    end
end

serialize_internal(r::RnnJLD,stackdict::IdDict)         = ((x,w) = rnninit(r.inputSize, r.hiddenSize, numLayers=r.numLayers, dropout=r.dropout,
                                                                           skipInput=(r.inputMode==1), bidirectional=(r.direction==1),
                                                                           rnnType=(:relu,:tanh,:lstm,:gru)[1+r.mode], algo=r.algo, dataType=r.dataType); x.w = serialize_internal(r.w,stackdict); x)
serialize_internal(x::Union{Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,Task},
                  stackdict::IdDict) = x
serialize_internal(x::Tuple, stackdict::IdDict) =
    ntuple(i->serialize_internal(x[i], stackdict), length(x))
serialize_internal(x::Module, stackdict::IdDict) = error("serialize of Modules not supported")

function serialize_internal(x::Core.SimpleVector, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    y = Core.svec(Any[serialize_internal(x[i], stackdict) for i = 1:length(x)]...)
    stackdict[x] = y
    return y
end

function serialize_internal(x::String, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    y = x
    stackdict[x] = y
    return y
end

function serialize_internal(@nospecialize(x), stackdict::IdDict)
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
                  serialize_internal(getfield(x,i), stackdict))
        end
    end
    return y::T
end

function serialize_internal(x::Array, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    _serialize_array_t(x, eltype(x), stackdict)
end

function _serialize_array_t(@nospecialize(x), T, stackdict::IdDict)
    if isbitstype(T)
        return (stackdict[x]=x)
    end
    y = map(xi->serialize_internal(xi,stackdict), x)
    stackdict[x] = y
    return y
end

function serialize_internal(x::Union{Dict,IdDict}, stackdict::IdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end

    if isbitstype(eltype(x))
        return (stackdict[x] = x)
    end

    dest = x <: Dict ? Dict() : IdDict()
    stackdict[x] = dest
    for (k, v) in x
        dest[serialize_internal(k, stackdict)] = serialize_internal(v, stackdict)
    end
    dest
end
