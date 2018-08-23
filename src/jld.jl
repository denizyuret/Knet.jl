using JLD2, FileIO

struct RnnJLD; inputSize; hiddenSize; numLayers; dropout; inputMode; direction; mode; algo; dataType; end
struct KnetJLD; a::Array ; end

function save(fname,args...;kwargs...)
     FileIO.save(fname,serialize.(args)...;kwargs...)
end
function load(fname,args...;kwargs...)
     serialize(FileIO.load(fname,args...;kwargs...))
end
# function load(fname;kwargs...)
#      serialize(FileIO.load(fname;kwargs...))
# end

macro save(filename, vars...)
    if isempty(vars)
        # Save all variables in the current module
        quote
            let
                m = $(__module__)
                f = jldopen($(esc(filename)), "w")
                wsession = JLD2.JLDWriteSession()
                try
                    for vname in names(m; all=true)
                        s = string(vname)
                        if !occursin(r"^_+[0-9]*$", s) # skip IJulia history vars
                            v = getfield(m, vname)
                            if !isa(v, Module)
                                try
                                    write(f, s, serialize(v), wsession)
                                catch e
                                    if isa(e, PointerException)
                                        @warn("skipping $vname because it contains a pointer")
                                    else
                                        rethrow(e)
                                    end
                                end
                            end
                        end
                    end
                finally
                    close(f)
                end
            end
        end
    else
        writeexprs = Vector{Expr}(undef, length(vars))
        for i = 1:length(vars)
            writeexprs[i] = :(write(f, $(string(vars[i])), serialize($(esc(vars[i]))), wsession))
        end

        quote
            jldopen($(esc(filename)), "w") do f
                wsession = JLD2.JLDWriteSession()
                $(Expr(:block, writeexprs...))
            end
        end
    end
end

macro load(filename, vars...)
    if isempty(vars)
        if isa(filename, Expr)
            throw(ArgumentError("filename argument must be a string literal unless variable names are specified"))
        end
        # Load all variables in the top level of the file
        readexprs = Expr[]
        vars = Symbol[]
        f = jldopen(filename)
        try
            for n in keys(f)
                if !JLD2.isgroup(f, JLD2.lookup_offset(f.root_group, n))
                    push!(vars, Symbol(n))
                end
            end
        finally
            close(f)
        end
    end
    return quote
        ($([esc(x) for x in vars]...),) = jldopen($(esc(filename))) do f
            ($([:(serialize(read(f, $(string(x))))) for x in vars]...),)
        end
        $(Symbol[v for v in vars]) # convert to Array
    end
end


serialize(x) = serialize_internal(x, IdDict())
serialize_internal(x::KnetArray,stackdict::IdDict)      = KnetJLD(Array(x))
serialize_internal(x::RNN,stackdict::IdDict)            = RnnJLD(x.inputSize, x.hiddenSize, x.numLayers, x.dropout, x.inputMode, x.direction, x.mode, x.algo, x.dataType)
serialize_internal(d::KnetJLD,stackdict::IdDict)        = (gpu() >= 0 ? KnetArray(d.a) : d.a)
serialize_internal(r::RnnJLD,stackdict::IdDict)         = rnninit(r.inputSize, r.hiddenSize, numLayers=r.numLayers, dropout=r.dropout,
                                                                  skipInput=(r.inputMode==1), bidirectional=(r.direction==1),
                                                                  rnnType=(:relu,:tanh,:lstm,:gru)[1+r.mode], algo=r.algo, dataType=r.dataType)[1]
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

    dest = Dict()
    stackdict[x] = dest
    for (k, v) in x
        dest[serialize_internal(k, stackdict)] = serialize_internal(v, stackdict)
    end
    dest
end
