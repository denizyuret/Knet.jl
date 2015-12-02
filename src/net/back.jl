"""
REWRITE:
back(r::Net,ygold,loss) computes the gradients of weights and
activations according to the given loss function and gold output.
ygold represents an individual item minibatch that may or may not be
an element of a sequence.  The seq keyword argument determines which:
initback sets incr=true for par if seq, back pops from stack if seq.
The loss gradient of the output, ygrad, is computed using
loss(ypred,ygold,ygrad).  ypred is retrieved from r.out[N] where N is
the index of the last op.  ygrad is written to r.dif[N].  If r.op[N]
has multiple outputs (toincr[N]), r.dif[N] is incremented.  If the
optional loss argument is not provided, ygold is used as the loss
gradient.  If ygold=nothing means the loss gradient from the output is
taken to be 0.  Gradient computation proceeds backwards from N..1.

"""

function back(f::Net, ygold=nothing, loss=copyloss; getdx=false, seq=false, o...)
    getdx = getdxbool(getdx, ninputs(f))
    initback(f, ygold, loss; getdx=getdx, seq=seq, o...) # TODO: fix initback: set :grad and :incr, on the reg or on the op? both on reg!
    gotreturn = false
    for n = length(f.prog):-1:1
        p = f.prog[n]
        get(p,:forw) || continue
        r = f.reg[p.output]
        if !gotreturn
            gotreturn = true
            p.output == :return || warn("Using $(p.output) as the return register.")
            if ygold == nothing
                get(r,:incr) || (r.dif = nothing)
            elseif r.dif0 == nothing
                # This may happen for a parameter free network
                @assert !any(q->(isa(q.op,Par)&&get(q,:forw)),f.prog)
            elseif get(r,:incr)
                loss(r.out, ygold, r.tmp; o...)
                r.dif = axpy!(1,r.tmp,r.dif0)
            else
                r.dif = loss(r.out, ygold, r.dif0; o...)
            end
        end
        inputregs = map(i->f.reg[i], p.inputs)
        if r.dif == nothing
            for ri in inputregs
                get(ri,:incr) || (ri.dif = nothing)
            end
        else
            dxn = Any[]
            xn = Any[]
            for ri in inputregs
                push!(dxn, !get(ri,:grad) ? nothing : get(ri,:incr) ? ri.tmp : ri.dif0)
                push!(xn, ri.out)
            end
            xn = get1(xn); yn = r.out; dyn = r.dif
            back(p.op, dyn, dxn...; x=xn, y=yn, o...)
            gpusync()
            for ri in inputregs
                if get(ri,:incr)
                    ri.dif = axpy!(1, ri.tmp, ri.dif0)
                    gpusync()
                elseif get(ri,:grad)
                    ri.dif = ri.dif0
                else
                    ri.dif = nothing
                end
            end
            gpusync()
            if get(r,:incr) && !isa(p.op, Par)
                # what if p.op=Arr?  then it will have no inputs, thus :grad=:incr=false
                # TODO: where does Par.dif get zeroed out?  reset?
                fill!(r.dif,0)
                gpusync()
            end
        end
        seq && get(p,:save) && pop(f,r)
    end
    if any(getdx)
        dx = Any[]; nx = 0
        for p in f.prog
            isa(p.op,Input) && getdx[nx+=1] && push!(dx, f.reg[p.output].dif);
        end
        return get1(dx)
    end
end

# this is used when no loss fn specified, in which case we assume ygold=ygrad
copyloss(ypred,ygold,ygrad;o...)=copy!(ygrad,ygold)

# turn various forms of getdx into boolean vector
function getdxbool(getdx, n)
    (isa(getdx, Vector{Bool}) && length(getdx)==n ? getdx :
     isa(getdx, Bool) ? fill(getdx, n) :
     isa(getdx, Vector{Int}) ? (tmp=falses(n);tmp[getdx]=true;tmp) :
     isa(getdx, Int) ? (tmp=falses(n);tmp[getdx]=true;tmp) :
     error("getdx=$getdx ninputs=$(n)"))
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
