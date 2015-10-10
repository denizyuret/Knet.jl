"""
back(r::Net,ygold; loss) for individual items that may or may not be elements
of a sequence.
"""
function back(r::Net, ygold, getdx...; loss=quadloss, seq=false, a...)
    N = nops(r)
    initback(r, ygold, getdx...; seq=seq, a...)
    if r.toincr[N] && ygold != nothing
        loss(r.out[N], ygold, r.tmp[N])
        r.dif[N] = axpy!(1,r.tmp[N],r.dif0[N])
    elseif ygold != nothing
        r.dif[N] = loss(r.out[N], ygold, r.dif0[N])
    elseif !r.toincr[N]
        r.dif[N] = nothing
    end
    for n = N:-1:1
        if r.dif[n] == nothing
            for i in r.inputs[n]
                !r.toincr[i] && (r.dif[i] = nothing)
            end
        else
            dxn = Any[]
            xn = Any[]
            for i in r.inputs[n]
                push!(dxn, !r.toback[i] ? nothing : r.toincr[i] ? r.tmp[i] : r.dif0[i])
                push!(xn, r.out[i]) 
            end
            xn = get1(xn); yn = r.out[n]; dyn = r.dif[n]
            back(r.op[n], dyn, dxn...; x=xn, y=yn, a...)
            for i in r.inputs[n]
                if r.toback[i]
                    r.toincr[i] && axpy!(1, r.tmp[i], r.dif0[i])
                    r.dif[i] = r.dif0[i]
                else
                    r.dif[i] = nothing
                end
            end
            r.toincr[n] && !isa(r.op[n], Par) && fill!(r.dif[n],0)
        end
        seq && r.tosave[n] && pop(r,n)
    end
    getdx==() || r.dif[find(o->isa(o,Input), r.op)]
end

### DEAD CODE:

# # back(r::Net,dy::Vector) for a sequence
# function back(r::Net, dy::Vector, dx...; a...)
#     dxi = map(x->(x==nothing ? x : x[end]), dx)
#     initback(r, dy[end], dxi...; seq=true, a...)
#     for i=length(dy):-1:1
#         dxi = map(x->(x==nothing ? x : x[i]), dx)
#         back(r, dy[i], dxi...; seq=true, a...)
#     end
# end

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

    # if dx != ()
    #     lastinput = 0
    #     for n = 1:N
    #         isa(r.op[n], Input) || continue
    #         copy!(dx[lastinput += 1], r.dif[n])
    #     end
    # end
