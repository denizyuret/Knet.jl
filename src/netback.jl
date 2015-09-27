# back(r::Net,dy::Vector) for a sequence
# TODO: truncated bptt
# - go forward k1 steps, run back for k2, update, recover state
# - if k1==k2 we just need the keepstate option to forw
# - if k1>k2 the stack won't be cleared
# - if k1<k2 the stack will be overdrawn

function back(r::Net, dy::Vector; dx=nothing, a...)
    r.dbg && display((:backseq0,length(dy),vecnorm0(params(r)))) # vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
    initback(r,dy[end]; seq=true, a...)
    for i=length(dy):-1:1
        dxi = (dx == nothing ? nothing : dx[i])
        back(r, dy[i]; seq=true, dx=dxi, a...)
    end
    r.dbg && display((:backseq1,length(dy),vecnorm0(params(r)))) # vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
end

# back(r::Net,dy) for individual items that may or may not be elements
# of a sequence.

function back(r::Net, dy; dx=nothing, seq=false, a...)
    r.dbg && display((:back0,seq,summary(dy),vecnorm0(params(r)))) # vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
    dx == nothing || length(dx) == ninputs(r) || error("Wrong number of inputs")
    seq || initback(r,dy; seq=false, a...)
    N = nops(r)
    if dy == nothing
        r.toincr[N] || (r.dif[N] = nothing)
    elseif eltype(dy) != eltype(r.dif0[N])
        error("Element type mismatch dy:$(eltype(dy)) dif0[$N]:$(eltype(r.dif0[N]))")
    elseif r.toincr[N]
        copy!(r.tmp[N], dy)
        r.dif[N] = axpy!(1,r.tmp[N],r.dif0[N])
    else
        r.dif[N] = copy!(r.dif0[N], dy)
    end										; dbg(r,:dif,N) 
    for n = N:-1:1
        if r.dif[n] == nothing
            for i in r.inputs[n]
                r.toincr[i] || (r.dif[i] = nothing)
            end
        else
            dxn = Any[]
            xn = Any[]
            for i in r.inputs[n]
                push!(dxn, r.toincr[i] ? arr(r.tmp[i]) : arr(r.dif0[i])) # TODO: use dx=nothing instead of returndx=false, that way we can choose which dx to return for multi-input case.
                push!(xn, arr(r.out[i])) # TODO: use path analysis to stop back/dx calculation for any path that does not lead to a parameter.
            end
            dxn = get1(dxn); xn = get1(xn); yn = arr(r.out[n]); dyn = arr(r.dif[n])
            back(r.op[n], dyn; incr=seq, x=xn, y=yn, dx=dxn, a...) # t:2164
            for i in r.inputs[n]
                r.toincr[i] && axpy!(1, r.tmp[i], r.dif0[i])            ; r.toincr[i]&&dbg(r,:tmp,i)
                r.dif[i] = r.dif0[i]                                    ; dbg(r,:dif,i)
            end
            r.toincr[n] && fill!(r.dif[n],0)                           ; r.toincr[n]&&dbg(r,:dif,n) # t:157
        end
        r.tosave[n] && pop(r,n)                                    ; r.tosave[n]&&dbg(r,:out,n)
    end
    for i = ninputs(r):-1:1
        n = i+N
        r.tosave[n] && pop(r,n)                                    ; r.tosave[n] && dbg(r,:out,n)
        dx == nothing || copy!(dx[i], r.dif[n])
    end
    r.dbg && display((:back1,seq,summary(dy),vecnorm0(params(r)))) # vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
    return dx
end

# initback(r::Net, dy) called at the beginning of a sequence or a
# stand-alone item, never between elements of a sequence.

function initback(r::Net, dy; seq=false, a...)
    r.dbg && display((:initback0,seq,summary(dy),vecnorm0(params(r))))  # vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
    fill!(r.dif, nothing)                           # why? (TODO)
    for n=1:length(r.dif0)
        y = (n==nops(r) && dy!=nothing ? dy : r.out0[n])
        initarray(r.dif0, n, y; dense=true) # x and dw may be sparse, dx and w always dense
        if r.toincr[n]
            initarray(r.tmp, n, r.dif0[n])
        end
    end
    for n=1:length(r.dif0)
        r.toincr[n] && fill!(r.dif0[n], 0)           # zeroed by back at every item in sequence
    end
    if seq
        for w in params(r)       # zeroed only once at the beginning of the sequence
            isdefined(w,:diff) && isdefined(w,:inc) && fill!(w.diff,0)
        end
    end
    r.dbg && display((:initback1,seq,summary(dy),vecnorm0(params(r)))) # vecnorm0(r.dif),vecnorm0(r.stack[1:r.sp])))
end

