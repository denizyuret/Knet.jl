# Logging functionality

export @debug, @trace, @logging_ccall, logging_run

# I/O without libuv, for use after STDOUT is finalized
raw_print(msg::AbstractString...) =
    ccall(:write, Cssize_t, (Cint, Cstring, Csize_t), 1, join(msg), length(join(msg)))
raw_println(msg::AbstractString...) = raw_print(msg..., "\n")

# safe version of `Base.print_with_color`, switching to raw I/O before finalizers are run
# (see `atexit` in `__init_logging__`)
const after_exit = Ref{Bool}(false)
function safe_print_with_color(color::Union{Int, Symbol}, io::IO, msg::AbstractString...)
    if after_exit[]
        raw_print(msg...)
    else
        print_with_color(color, io, msg...)
    end
end

# Display a trace message. Only results in actual printing if the TRACE environment variable
# is set.
const TRACE = haskey(ENV, "TRACE")
@inline function trace(io::IO, msg...; prefix="TRACE: ", line=true)
    @static if TRACE
        safe_print_with_color(:cyan, io, prefix, chomp(string(msg...)), line ? "\n" : "")
    end
end
@inline trace(msg...; kwargs...) = trace(STDERR, msg...; kwargs...)
macro trace(args...)
    TRACE && return Expr(:call, :trace, map(esc, args)...)
end

# Display a debug message. Only results in actual printing if the TRACE or DEBUG environment
# variable is set.
const DEBUG = TRACE || haskey(ENV, "DEBUG")
@inline function debug(io::IO, msg...; prefix="DEBUG: ", line=true)
    @static if DEBUG
        safe_print_with_color(:green, io, prefix, chomp(string(msg...)), line ? "\n" : "")
    end
end
@inline debug(msg...; kwargs...) = debug(STDERR, msg...; kwargs...)
macro debug(args...)
    DEBUG && return Expr(:call, :debug, map(esc, args)...)
end

# Create an indented string from any value (instead of escaping endlines as \n)
function repr_indented(ex; prefix=" "^7, abbrev=true)
    io = IOBuffer()
    print(io, ex)
    str = String(take!(io))

    # Limit output
    if abbrev && length(str) > 256
        if isa(ex, Array)
            T = eltype(ex)
            dims = join(size(ex), " by ")
            if method_exists(zero, (T,)) && zeros(ex) == ex
                str = "$T[$dims zeros]"
            else
                str = "$T[$dims elements]"
            end
        else
            if contains(strip(str), "\n")
                str = str[1:100] * "…\n\n[snip]\n\n…" * str[end-100:end]
            else
                str = str[1:100] * "…" * str[end-100:end]
            end
        end
    end

    lines = split(strip(str), '\n')
    if length(lines) > 1
        for i = 1:length(lines)
            lines[i] = prefix * lines[i]
        end

        lines[1] = "\"\n" * lines[1]
        lines[length(lines)] = lines[length(lines)] * "\""

        return join(lines, '\n')
    else
        return str
    end
end


# ccall wrapper logging the call, its arguments, and the returned value.
# Only logs if TRACE environment variable is set.
macro logging_ccall(fun, target, rettyp, argtypes, args...)
    blk = Expr(:block)

    # print the function name & arguments
    if TRACE
        push!(blk.args, :(trace($(sprint(Base.show_unquoted,fun.args[1])*"("); line=false)))
        i = length(args)
        for arg in args
            i -= 1
            sep = (i>0 ? ", " : "")

            # TODO: we should only do this if evaluating `arg` has no side effects
            push!(blk.args, :(trace(repr_indented($(esc(arg))), $sep;
                  prefix=$(sprint(Base.show_unquoted,arg))*"=", line=false)))
        end
        push!(blk.args, :(trace(""; prefix=") =", line=false)))
    end

    # actual ccall
    @gensym ret
    push!(blk.args, quote
        $ret = ccall($(esc(target)), $(esc(rettyp)), $(esc(argtypes)), $(map(esc, args)...))
    end)

    # print results
    if TRACE
        push!(blk.args, :(trace($ret; prefix=" ")))
    end

    push!(blk.args, :($ret))

    return blk
end

function logging_run(cmd)
    if TRACE
        println(cmd)
    end
    run(cmd)
end


function __init_logging__()
    if TRACE
        trace("CUDA packages running in trace mode, this will generate a lot of additional output")
    elseif DEBUG
        debug("CUDA packages running in debug mode, this will generate additional output")
        debug("Run with TRACE=1 to enable even more output")
    end

    atexit(()->begin
        debug("Dropping down to post-finalizer I/O")
        after_exit[]=true
    end)
end
