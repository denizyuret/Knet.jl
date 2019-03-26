using JLD2, FileIO

"""
    Knet.save(filename, args...; kwargs...)

Call `FileIO.save` after serializing Knet specific args. 

File format is determined by the filename extension. JLD and JLD2 are supported. Other formats
may work if supported by FileIO, please refer to the documentation of FileIO and the specific
format.  Example:

    Knet.save("foo.jld2", "name1", value1, "name2", value2)
"""
function save(fname,args...;kwargs...)
     FileIO.save(fname,serialize.(args)...;kwargs...)
end

"""
    Knet.load(filename, args...; kwargs...)

Call `FileIO.load` then deserialize Knet specific values.

File format is determined by FileIO. JLD and JLD2 are supported. Other formats may work if
supported by FileIO, please refer to the documentation of FileIO and the specific format.
Example:

    Knet.load("foo.jld2")           # returns a ("name"=>value) dictionary
    Knet.load("foo.jld2", "name1")  # returns the value of "name1" in "foo.jld2"
    Knet.load("foo.jld2", "name1", "name2")   # returns tuple (value1, value2)
"""
function load(fname,args...;kwargs...)
     serialize(FileIO.load(fname,args...;kwargs...))
end

"""
    Knet.@save "filename" variable1 variable2...

Save the values of the specified variables to filename in JLD2 format.

When called with no variable arguments, write all variables in the global scope of the current
module to filename.  See [JLD2](https://github.com/JuliaIO/JLD2.jl).
"""
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

"""
    Knet.@load "filename" variable1 variable2...

Load the values of the specified variables from filename in JLD2 format.

When called with no variable arguments, load all variables in filename.  See
[JLD2](https://github.com/JuliaIO/JLD2.jl).
"""
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
