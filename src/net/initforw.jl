"""
initforw sets r.tosave, r.out, r.out0 used by forw.
possibly r.dif0[N] (if loss calculated).
r.tosave is read-only, used if seq.
"""
function initforw(r::Net, inputs...; keepstate=false, ygold=nothing, seq=false, o...)
    @assert length(inputs) == ninputs(r)
    @assert r.sp == 0
    isassigned(r.out0, 1) || initforw0(r, inputs...)
    N = length(r.op)
    lastinput = 0
    for n=1:N
        if isa(r.op[n], Input)
            i = inputs[lastinput += 1]
            o = r.out0[n]
            @assert issimilar3(i, o)
        end
    end
    if ygold != nothing
        @assert issimilar2(ygold, r.out0[N])
        if isassigned(r.dif0, N)
            @assert issimilar3(ygold, r.dif0[N])
        else
            r.dif0[N] = newarray(gpu(), stype(ygold), eltype(ygold), size(ygold))
        end
    end
    if !seq
        # @assert all((r.out .== nothing) | (r.out .== r.out0)) # t:110/244
        @assert !keepstate "meaningless keepstate in non-sequence run"
        fill!(r.out, nothing)
    elseif keepstate
        copy!(r.out, r.out0)
    else
        # @assert all((r.out .== nothing) | (r.out .== r.out0))
        fill!(r.out, nothing)
    end
    # TODO: implement batch size changes
end

# first init: infer and alloc
function initforw0(r::Net, inputs...)
    xtype = infertype(r, inputs...)
    sizes = infersize(r, inputs...)
    lastinput = 0
    for n=1:length(r.op)
        ### REGISTER SHARING OPTIMIZATION:
        st = (isa(r.op[n], Input) ? stype(inputs[lastinput += 1]) : nothing)
        r.out0[n] = findout(r, n, sizes, st)
        if r.out0[n] == nothing
            r.out0[n] = newarray(gpu(), st, xtype, sizes[n])
        end
    end
    fill!(r.out, nothing)
    # TODO: figure out tmp
end

function findout(r::Net, n, sizes, nsparse)
    r.tosave[n] && return        # saved regs and pars should not overwrite or be overwritten
    isa(r.op[n], Par) && return  # TODO: how about rnd and con?
    free = nothing               # search most recent written first to avoid copying in overwriting ops
    for i = n-1:-1:1             # considering overwriting i with n
        r.tosave[i] && continue
        isa(r.op[i], Par) && continue
        size(r.out0[i]) == sizes[n] || continue
        stype(r.out0[i]) == nsparse || continue
        !overwrites(r.op[n]) && in(i, r.inputs[n]) && continue
        willberead = false                              # is anybody going to read i before it is written again?
        k = n
        while true
            k = mod1(k+1, length(r.op))
            for j in r.inputs[k]
                isassigned(r.out0, j) && r.out0[j] === r.out0[i] && (willberead = true; break)
            end
            willberead && break
            isassigned(r.out0,k) && r.out0[k] === r.out0[i] && break
        end
        !willberead && (free = r.out0[i]; break)
    end
    return free
end

function infersize(r::Net, inputs...)
    N = length(r.op)
    dims = fill!(cell(N), nothing)
    lastinput = 0
    notfound = N
    while notfound > 0
        for n=1:N
            if isa(r.op[n], Input)
                dims[n] == nothing && (dims[n] = size(inputs[lastinput += 1]))
            else
                d = infersize(r.op[n], dims[r.inputs[n]]...)
                d == nothing && continue
                dims[n] = d[end]
                dims[r.inputs[n]] = [d[1:end-1]...]
            end
        end
        nf = count(x->(x==nothing || prod(x)==0), dims)
        nf == notfound && error("Cannot infer sizes: $dims")
        notfound = nf
    end
    return dims
end

function infertype(r::Net, inputs...)
    it = nothing
    for i in inputs
        t = eltype(i)
        if it == nothing
            it = t
        elseif t != it
            error("Conflicting input eltypes")
        end
    end
    it == nothing && error("Cannot infer eltype")
    return it
    # TODO: deal with inputless networks:
end

