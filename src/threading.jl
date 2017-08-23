#=
An abstraction over Julia's experimental threading library
for version compatibility
=#


nthreads() = Threads.nthreads()
threadid() = Threads.threadid()

# Thread-safe printing support
# tsprint & tsprintln register hooks to print
# tprint prints all hooks thread by thread

"""Uncomment here after figure more debugging"""
#=let buffer = Vector{Any}(nthreads())
    global tsprint, tsprintln, tprint

    reset() = for i = 1:length(buffer); buffer[i] = []; end
    reset()
    
    function _tsprint(pfn, args...)
        push!(buffer[threadid()], (pfn, args))
    end

    tsprint(args...) = _tsprint(print, args...)

    tsprintln(args...) = _tsprint(println, args...)

    function tprint(;reset_buffer=true)
        for i = 1:nthreads()
            println("\nThread ", i, ":\n------")
            for (fn, data) in  buffer[i]
                fn(data...)
            end
        end
        if reset_buffer; reset(); end
    end
end=#

#=
threads(1:2) do t
...
end
=#
function threads(f::Function, range::Union{Range, Void}=nothing)
    if range == nothing
        range = 1:nthreads()
    end
    Threads.@threads for t = range
        f(t)
    end
end
