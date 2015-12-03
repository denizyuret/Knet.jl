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

function back(f::Net, ygold=nothing, loss=copyloss; getdx=false, o...)
    getdx = getdxbool(getdx, ninputs(f))
    initback(f, ygold, loss, getdx) # TODO: rethink lastforw==lastback in a seq context

    yreg = getreg(f, :return)
    if yreg != nothing
        if ygold == nothing && get(yreg,:incr) # TODO: fix incr to take into account :return status
            # nothing to do
        elseif ygold == nothing
            yreg.dif = nothing
        elseif get(yreg,:incr)
            loss(yreg.out, ygold, yreg.tmp; o...) # TODO: are we sure yreg.out has not been changed?
            yreg.dif = axpy!(1,yreg.tmp,yreg.dif0)
        else
            yreg.dif = loss(yreg.out, ygold, yreg.dif0; o...)
        end
    elseif ygold != nothing
        error("ygold specified when there is no output")
    end

    for n = length(f.prog):-1:1
        s = pop!(f.stack)
        s == nothing && continue
        (p, xsave, ysave) = s
        @assert p === f.prog[n]
        y = output_register(f,p)
        get(y,:grad) || continue
        global xx = input_registers(f,p)
        if y.dif == nothing
            for x in xx
                get(x,:grad) && !get(x,:incr) && (x.dif = nothing)
            end
        else
            dxx = map(x->(!get(x,:grad) ? nothing : get(x,:incr) ? x.tmp : x.dif0), xx)
            back(p.op, y.dif, dxx...; x=get1(xsave), y=ysave, o...)
            gpusync()
            for x in xx
                x.dif = (get(x,:incr) ? axpy!(1, x.tmp, x.dif0) :
                         get(x,:grad) ? x.dif0 : nothing)
                gpusync()
            end
            if get(y,:incr) && !isa(p.op, Par)
                # what if p.op=Arr?  then it will have no inputs, thus :grad=:incr=false
                # where does Par.dif get zeroed out? at reset!
                @show p
                fill!(y.dif,0)
                gpusync()
            end
        end
    end
    if any(getdx)
        dx = Any[]; nx = 0
        for p in f.prog
            isa(p.op,Input) && getdx[nx+=1] && push!(dx, getdif(f,p))
        end
        return get1(dx)
    end
end

# this is used when no loss fn specified, in which case we assume ygold is actually ygrad
copyloss(ypred,ygold,ygrad;o...)=copy!(ygrad,ygold)

# turn various forms of getdx into boolean vector
function getdxbool(getdx, n)
    (isa(getdx, Vector{Bool}) && length(getdx)==n ? getdx :
     isa(getdx, Bool) ? fill(getdx, n) :
     isa(getdx, Vector{Int}) ? (tmp=falses(n);tmp[getdx]=true;tmp) :
     isa(getdx, Int) ? (tmp=falses(n);tmp[getdx]=true;tmp) :
     error("getdx=$getdx ninputs=$(n)"))
end

get1(x)=(!isempty(methods(length, (typeof(x),))) && length(x)==1?x[1]:x)

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
