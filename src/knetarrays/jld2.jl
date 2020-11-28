import JLD2, FileIO

# With the following standard FileIO.save, FileIO.load, JLD2.@save, JLD2.@load should work
struct JLD2KnetArray{T,N}; array::Array{T,N}; end
JLD2.writeas(::Type{KnetArray{T,N}}) where {T,N} = JLD2KnetArray{T,N}
JLD2.wconvert(::Type{JLD2KnetArray{T,N}}, x::KnetArray{T,N}) where {T,N} = JLD2KnetArray(Array(x))
JLD2.rconvert(::Type{KnetArray{T,N}}, x::JLD2KnetArray{T,N}) where {T,N} = KnetArray(x.array)


# These are deprecated functions and macros for backward compatibility and loading old files

function save(file, args...; options...)
    @warn "Knet.save is deprecated, please use FileIO.save/load instead" maxlog=1
    FileIO.save(file, jld2serialize.(args)...; options...)
end

function load(file, args...; options...)
    @warn "Knet.load is deprecated, please use FileIO.save/load instead" maxlog=1
    jld2serialize(FileIO.load(file, args...; options...))
end


"""
    Knet.@save "filename" variable1 variable2...
Save the values of the specified variables to filename in JLD2 format.
When called with no variable arguments, write all variables in the global scope of the current
module to filename.  See [JLD2](https://github.com/JuliaIO/JLD2.jl).

This macro is deprecated, please use `JLD2.@save` instead.
"""
macro save(filename, vars...)
    if isempty(vars)
        # Save all variables in the current module
        quote
            @warn "Knet.@save is deprecated, please use JLD2.@save/@load instead" maxlog=1
            let
                m = $(__module__)
                f = JLD2.jldopen($(esc(filename)), "w")
                wsession = JLD2.JLDWriteSession()
                try
                    for vname in names(m; all=true)
                        s = string(vname)
                        if !occursin(r"^_+[0-9]*$", s) # skip IJulia history vars
                            v = getfield(m, vname)
                            if !isa(v, Module)
                                try
                                    write(f, s, jld2serialize(v), wsession)
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
            writeexprs[i] = :(write(f, $(string(vars[i])), jld2serialize($(esc(vars[i]))), wsession))
        end

        quote
            @warn "Knet.@save is deprecated, please use JLD2.@save/@load instead" maxlog=1
            JLD2.jldopen($(esc(filename)), "w") do f
                wsession = JLD2.JLDWriteSession()
                $(Expr(:block, writeexprs...))
            end
        end
    end
end

"""
    Knet.@load "filename" variable1 variable2...
Load the values of the specified variables from filename in JLD2 format.
When called with no variable arguments, load all variables in filename.  See
[JLD2](https://github.com/JuliaIO/JLD2.jl).

This macro is deprecated, please use `JLD2.@load` instead.
"""
macro load(filename, vars...)
    if isempty(vars)
        if isa(filename, Expr)
            throw(ArgumentError("filename argument must be a string literal unless variable names are specified"))
        end
        # Load all variables in the top level of the file
        readexprs = Expr[]
        vars = Symbol[]
        f = JLD2.jldopen(filename)
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
        @warn "Knet.@load is deprecated, please use JLD2.@save/@load instead" maxlog=1
        ($([esc(x) for x in vars]...),) = JLD2.jldopen($(esc(filename))) do f
            ($([:(jld2serialize(read(f, $(string(x))))) for x in vars]...),)
        end
        $(Symbol[v for v in vars]) # convert to Array
    end
end
