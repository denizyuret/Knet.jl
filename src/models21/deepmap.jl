# Recursively search for type T and perform an operation on it based on julia/base/deepcopy.jl

function deepmap(f, T, x)
    @assert !isbitstype(T) "deepmap does not support isbitstype(T)"
    deepmap_internal(f, T, Set{UInt}(), x)
end

function deepmap_internal(f, T, s::Set{UInt}, x::Union{Module,Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,UnionAll,Task,Regex})
    if x isa T && objectid(x) ∉ s
        push!(s, objectid(x))
        f(x)
    end
end

function deepmap_internal(f, T, s::Set{UInt}, x::Union{Tuple,Core.SimpleVector})
    objectid(x) ∈ s && return
    push!(s, objectid(x))
    x isa T && f(x)
    for xi in x
        !isbits(xi) && deepmap_internal(f, T, s, xi)
    end
end

function deepmap_internal(f, T, s::Set{UInt}, x::String)
    objectid(x) ∈ s && return
    push!(s, objectid(x))
    x isa T && f(x)
end

function deepmap_internal(f, T, s::Set{UInt}, @nospecialize(x))
    X = typeof(x)::DataType
    isbitstype(X) && return
    objectid(x) ∈ s && return
    push!(s, objectid(x))
    x isa T && f(x)
    for i in 1:nfields(x)
        if isdefined(x, i)
            xi = getfield(x,i)
            !isbits(xi) && deepmap_internal(f, T, s, xi)
        end
    end
end

function deepmap_internal(f, T, s::Set{UInt}, x::Array)
    objectid(x) ∈ s && return
    push!(s, objectid(x))
    x isa T && f(x)
    isbitstype(eltype(x)) && return
    for i = 1:length(x)
        if ccall(:jl_array_isassigned, Cint, (Any, Csize_t), x, i-1) != 0
            xi = ccall(:jl_arrayref, Any, (Any, Csize_t), x, i-1)
            !isbits(xi) && deepmap_internal(f, T, s, xi)
        end
    end
end

function deepmap_internal(f, T, s::Set{UInt}, x::Union{Dict,IdDict})
    objectid(x) ∈ s && return
    push!(s, objectid(x))
    x isa T && f(x)
    isbitstype(eltype(x)) && return
    for (k, v) in x
        !isbits(k) && deepmap_internal(f, T, s, k)
        !isbits(v) && deepmap_internal(f, T, s, v)
    end
end
