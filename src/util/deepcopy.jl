# deep copying
# This is copied from base/deepcopy.jl
# Duplicated to cpucopy and gpucopy to facilitate cpu/gpu copying.
# New types can override cpucopy_internal and gpucopy_internal.

using CUDArt
using Base: arrayset

cpucopy(x) = cpucopy_internal(x, ObjectIdDict())

cpucopy_internal(x::Union(Symbol,LambdaStaticData,TopNode,QuoteNode,DataType,UnionType,Task), stackdict::ObjectIdDict) = x
cpucopy_internal(x::Tuple, stackdict::ObjectIdDict) = ntuple(length(x), i->cpucopy_internal(x[i], stackdict))
cpucopy_internal(x::Module, stackdict::ObjectIdDict) = error("cpucopy of Modules not supported")

function cpucopy_internal(x::Function, stackdict::ObjectIdDict)
    if isa(x.env, Union(MethodTable, Symbol)) || x.env === ()
        return x
    end
    invoke(cpucopy_internal, Tuple{Any, ObjectIdDict}, x, stackdict)
end

function cpucopy_internal(x, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    _cpucopy_t(x, typeof(x), stackdict)
end

function _cpucopy_t(x, T::DataType, stackdict::ObjectIdDict)
    nf = nfields(T)
    (isbits(T) || nf == 0) && return x
    if T.mutable
        y = ccall(:jl_new_struct_uninit, Any, (Any,), T)
        stackdict[x] = y
        for i in 1:nf
            if isdefined(x,i)
                y.(i) = cpucopy_internal(x.(i), stackdict)
            end
        end
    else
        fields = Any[cpucopy_internal(x.(i), stackdict) for i in 1:nf]
        y = ccall(:jl_new_structv, Any, (Any, Ptr{Void}, UInt32),
                  T, pointer(fields), length(fields))
    end
    return y::T
end

function cpucopy_internal(x::Array, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    _cpucopy_array_t(x, eltype(x), stackdict)
end

# CUDA extension:
function cpucopy_internal(x::CudaArray, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    to_host(x)
end

function _cpucopy_array_t(x, T, stackdict::ObjectIdDict)
    if isbits(T)
        return copy(x)
    end
    dest = similar(x)
    stackdict[x] = dest
    for i=1:length(x)
        if isdefined(x,i)
            arrayset(dest, cpucopy_internal(x[i], stackdict), i)
        end
    end
    return dest
end

gpucopy(x) = gpucopy_internal(x, ObjectIdDict())

gpucopy_internal(x::Union(Symbol,LambdaStaticData,TopNode,QuoteNode,DataType,UnionType,Task), stackdict::ObjectIdDict) = x
gpucopy_internal(x::Tuple, stackdict::ObjectIdDict) = ntuple(length(x), i->gpucopy_internal(x[i], stackdict))
gpucopy_internal(x::Module, stackdict::ObjectIdDict) = error("gpucopy of Modules not supported")

function gpucopy_internal(x::Function, stackdict::ObjectIdDict)
    if isa(x.env, Union(MethodTable, Symbol)) || x.env === ()
        return x
    end
    invoke(gpucopy_internal, Tuple{Any, ObjectIdDict}, x, stackdict)
end

function gpucopy_internal(x, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    _gpucopy_t(x, typeof(x), stackdict)
end

function _gpucopy_t(x, T::DataType, stackdict::ObjectIdDict)
    nf = nfields(T)
    (isbits(T) || nf == 0) && return x
    if T.mutable
        y = ccall(:jl_new_struct_uninit, Any, (Any,), T)
        stackdict[x] = y
        for i in 1:nf
            if isdefined(x,i)
                y.(i) = gpucopy_internal(x.(i), stackdict)
            end
        end
    else
        fields = Any[gpucopy_internal(x.(i), stackdict) for i in 1:nf]
        y = ccall(:jl_new_structv, Any, (Any, Ptr{Void}, UInt32),
                  T, pointer(fields), length(fields))
    end
    return y::T
end

function gpucopy_internal(x::Array, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    _gpucopy_array_t(x, eltype(x), stackdict)
end

# CUDA extension:
function gpucopy_internal(x::CudaArray, stackdict::ObjectIdDict)
    if haskey(stackdict, x)
        return stackdict[x]
    end
    copy(x)
end

function _gpucopy_array_t(x, T, stackdict::ObjectIdDict)
    if isbits(T)
        # CUDA extension:
        return CudaArray(x)
    end
    dest = similar(x)
    stackdict[x] = dest
    for i=1:length(x)
        if isdefined(x,i)
            arrayset(dest, gpucopy_internal(x[i], stackdict), i)
        end
    end
    return dest
end
