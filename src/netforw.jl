"""
forw(r::Net,x::Vector) for sequence.
x can be a Vector of Arrays representing items.
x can be a Vector of Tuples representing multiple inputs.
x cannot be a Vector of scalars (TODO:think this over)
"""
function forw(r::Net, x::Vector; yout=nothing, ygold=nothing, a...)
    DBG && display((:forwseq0,length(x),vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    isbits(eltype(x)) && error("forw expects a minibatch")
    x1 = (isa(x[1], Tuple) ? x[1] : (x[1],))
    initforw(r, x1...; a...)
    loss = 0.0
    for i=1:length(x)
        xi = (isa(x[i], Tuple) ? x[i] : (x[i],))
        yi = (yout == nothing ? nothing : yout[i])
        yg = (ygold == nothing ? nothing : ygold[i])
        loss += forw(r, xi...; seq=true, yout=yi, ygold=yg, a...)
    end
    DBG && display((:forwseq1,length(x),vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    return loss
end

# forw(r::Net,x...) for individual items (or item tuples) that may or
# may not be part of a sequence.

function forw(r::Net, inputs...; yout=nothing, ygold=nothing, seq=false, trn=false, a...)
    length(inputs) == ninputs(r) || error("Wrong number of inputs")
    seq || initforw(r, inputs...; a...)
    DBG && display((:forw0,seq,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    N = nops(r)
    lastinput = 0
    for n = 1:N
        trn && r.tosave[n] && push(r,n)         # t:327
        if isa(r.op[n], Input)                       # TODO: forw.op interface differences: returning y, train/trn, y/yout, ...
            r.out[n] = copy!(r.out0[n], inputs[lastinput += 1])
        else
            r.out[n] = forw(r.op[n], r.out[r.inputs[n]]..., r.out0[n]; train=trn, a...) # ;dbg(r,:out,n) # t:2300
        end
    end
    DBG && display((:forw1,seq,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    yout != nothing && copy!(yout, r.out[N])
    if ygold != nothing
        initarray(r.dif0, N, ygold)
        r.dif[N] = copy!(r.dif0[N], ygold)
        return loss(r.op[N], r.dif[N]; y=r.out[N])
    else
        return 0.0
    end
end
