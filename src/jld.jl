using JLD2, FileIO

function save(fname,args...;kwargs...)
     FileIO.save(fname,serialize.(args)...;kwargs...)
end
function load(fname,args...;kwargs...)
     serialize(FileIO.load(fname,args...;kwargs...))
end

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
