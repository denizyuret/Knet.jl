"""

back(r::Net,ygold,loss) for individual items that may or may not be
elements of a sequence.  The seq keyword argument determines which:
initback sets incr=true for par if seq, back pops from stack if seq.
The loss gradient of the output, ygrad, is computed using
loss(ypred,ygold,ygrad).  ypred is retrieved from r.out[N] where N is
the index of the last op.  ygrad is written to r.dif[N].  If r.op[N]
has multiple outputs (toincr[N]), r.dif[N] is incremented.  If the
optional loss argument is not provided, ygold is used as the loss
gradient.  If ygold=nothing means the loss gradient from the output is
taken to be 0.  Gradients computation proceeds backwards from N..1.

"""
function back(r::Net, ygold=nothing, loss=copyloss; getdx=false, seq=false, o...)
    N = nops(r)
    initback(r, ygold, loss; getdx=getdx, seq=seq, o...)
    if ygold == nothing
        r.toincr[N] || (r.dif[N] = nothing)
    elseif r.dif0[N] == nothing
        # This may happen for a parameter free network
        @assert !any(r.toback)
    elseif !r.toincr[N]
        r.dif[N] = loss(r.out[N], ygold, r.dif0[N]; o...)
    else
        loss(r.out[N], ygold, r.tmp[N]; o...)
        r.dif[N] = axpy!(1,r.tmp[N],r.dif0[N])
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
            back(r.op[n], dyn, dxn...; x=xn, y=yn, o...)
            gpusync()
            for i in r.inputs[n]
                if r.toback[i]
                    if r.toincr[i]
                        axpy!(1, r.tmp[i], r.dif0[i]) 
                        gpusync()
                    end
                    r.dif[i] = r.dif0[i]
                else
                    r.dif[i] = nothing
                end
            end
            gpusync()
            if r.toincr[n] && !isa(r.op[n], Par)
                fill!(r.dif[n],0)
                gpusync()
            end
        end
        seq && r.tosave[n] && pop(r,n)
    end
    getdx && get1(r.dif[find(op->isa(op,Input), r.op)])
end

copyloss(ypred,ygold,ygrad;o...)=copy!(ygrad,ygold)

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
