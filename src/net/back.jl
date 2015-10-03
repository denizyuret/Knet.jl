"""
back(r::Net,dy,dx...) for individual items that may or may not be elements
of a sequence.
"""
function back(r::Net, dy, dx...; seq=false, a...)
    N = nops(r)
    seq || initback(r, dy, dx...; seq=false, a...)
    @assert dy == nothing || issimilar2(dy, r.dif0[N])
    if dy == nothing
        r.toincr[N] || (r.dif[N] = nothing) # otherwise we pretend we add 0
    elseif r.toincr[N]
        copy!(r.tmp[N], dy)
        r.dif[N] = axpy!(1,r.tmp[N],r.dif0[N])
    else
        r.dif[N] = copy!(r.dif0[N], dy)
    end										# ; dbg(r,:dif,N) 
    for n = N:-1:1
        if r.dif[n] == nothing
            for i in r.inputs[n]
                r.toincr[i] || (r.dif[i] = nothing)
            end
        else
            dxn = Any[]
            xn = Any[]
            for i in r.inputs[n]
                push!(dxn, !r.toback[i] ? nothing : r.toincr[i] ? r.tmp[i] : r.dif0[i])
                push!(xn, r.out[i]) 
            end
            xn = get1(xn); yn = r.out[n]; dyn = r.dif[n]
            back(r.op[n], dyn, dxn...; x=xn, y=yn, a...) # t:2164
            for i in r.inputs[n]
                if r.toback[i]
                    r.toincr[i] && axpy!(1, r.tmp[i], r.dif0[i])            # ; r.toincr[i]&&dbg(r,:tmp,i)
                    r.dif[i] = r.dif0[i]                                    # ; dbg(r,:dif,i)
                else
                    r.dif[i] = nothing
                end
            end
            r.toincr[n] && !isa(r.op[n], Par) && fill!(r.dif[n],0)
        end
        seq && r.tosave[n] && pop(r,n)                                    # ; r.tosave[n]&&dbg(r,:out,n)
    end
    if dx != ()
        lastinput = 0
        for n = 1:N
            isa(r.op[n], Input) || continue
            copy!(dx[lastinput += 1], r.dif[n])
        end
    end
end

# back(r::Net,dy::Vector) for a sequence
function back(r::Net, dy::Vector, dx...; a...)
    dxi = map(x->(x==nothing ? x : x[end]), dx)
    initback(r, dy[end], dxi...; seq=true, a...)
    for i=length(dy):-1:1
        dxi = map(x->(x==nothing ? x : x[i]), dx)
        back(r, dy[i], dxi...; seq=true, a...)
    end
end

# DONE: truncated bptt
# - go forward k1 steps, run back for k2, update, recover state
# - if k1==k2 we just need the keepstate option to forw
# - if k1>k2 the stack won't be cleared
# - if k1<k2 the stack will be overdrawn

    # for i = ninputs(r):-1:1
    #     n = i+N
    #     r.tosave[n] && pop(r,n)                                    # ; r.tosave[n] && dbg(r,:out,n)
    #     dx == nothing || copy!(dx[i], r.dif[n])
    # end
