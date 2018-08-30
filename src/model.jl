abstract type Model end

param(d...; init=xavier, atype=(gpu() >= 0 ? KnetArray{Float32} : Array{Float32}))=Param(atype(init(d...)))

# We should take care of iterating over parameters automatically:
# So `for param in model` works for any model.

Base.iterate(f::Model, s=(params(f),1)) = ((p,i)=s; i<=length(p) ? (p[i],(p,i+1)) : nothing)

params(f::Model) = (ps=Param[]; params_internal(f,ps,IdDict()); ps)
params_internal(p::Param, ps::Vector{Param}, d::IdDict) = if !haskey(d,p); d[p]=true; push!(ps,p); end

# Based on deepcopy_internal:

params_internal(x::Union{Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,Task},
                ps::Vector{Param}, stackdict::IdDict) = return
params_internal(x::Tuple, ps::Vector{Param}, stackdict::IdDict) =
    for p in x; params_internal(p, ps, stackdict); end
params_internal(x::Module, ps::Vector{Param}, stackdict::IdDict) = return

function params_internal(x::Core.SimpleVector, ps::Vector{Param}, stackdict::IdDict)
    if haskey(stackdict, x)
        return
    end
    stackdict[x] = true
    for p in x; params_internal(x, ps, stackdict); end
end

params_internal(x::String, ps::Vector{Param}, stackdict::IdDict) = return

function params_internal(@nospecialize(x), ps::Vector{Param}, stackdict::IdDict)
    T = typeof(x)::DataType
    nf = nfields(x)
    (isbitstype(T) || nf == 0) && return
    if haskey(stackdict, x)
        return
    end
    if T.mutable
        stackdict[x] = true
    end
    for i in 1:nf
        if isdefined(x,i)
            params_internal(getfield(x,i), ps, stackdict)
        end
    end
end

function params_internal(x::Array, ps::Vector{Param}, stackdict::IdDict)
    if haskey(stackdict, x)
        return
    end
    _params_array_t(x, eltype(x), ps, stackdict)
end

function _params_array_t(@nospecialize(x), T, ps::Vector{Param}, stackdict::IdDict)
    stackdict[x] = true
    if isbitstype(T)
        return
    end
    for i = 1:(length(x)::Int)
        if ccall(:jl_array_isassigned, Cint, (Any, Csize_t), x, i-1) != 0
            xi = ccall(:jl_arrayref, Any, (Any, Csize_t), x, i-1)
            if !isbits(xi)
                xi = params_internal(xi, ps, stackdict)
            end
        end
    end
end

function params_internal(x::Union{Dict,IdDict}, ps::Vector{Param}, stackdict::IdDict)
    if haskey(stackdict, x)
        return
    end
    stackdict[x] = true
    for (k, v) in x
        params_internal(k, ps, stackdict)
        params_internal(v, ps, stackdict)
    end
end
