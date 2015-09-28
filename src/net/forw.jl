"""
forw(r::Net,x...) for individual items (or item tuples) that may or
may not be part of a sequence.
TODO: make yout optional last argument to match the op interface?
"""
function forw(r::Net, inputs...; ygold=nothing, seq=false, mode=:test, a...)
    if length(inputs) == ninputs(r)
        yout = nothing
    elseif length(inputs) == 1+ninputs(r)
        yout = inputs[end]
        inputs = inputs[1:end-1]
    else
        error("Wrong number of inputs")
    end
    seq || initforw(r, inputs...; ygold=ygold, seq=false, a...)
    # display((:forw0,seq,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    N = nops(r)
    lastinput = 0
    for n = 1:N
        mode==:train && seq && r.tosave[n] && push(r,n) # t:327
        x = r.out[r.inputs[n]]
        if isa(r.op[n], Input)
            r.out[n] = copy!(r.out0[n], inputs[lastinput += 1])
        else
            r.out[n] = forw(r.op[n], x..., r.out0[n]; mode=mode, a...) # ;dbg(r,:out,n) # t:2300
        end
    end
    yout != nothing && copy!(yout, r.out[N])
    loss1 = 0.0
    if ygold != nothing
        r.dif[N] = copy!(r.dif0[N], ygold)
        loss1 = loss(r.op[N], r.dif[N], r.out[N])
    end
    # display((:forw1,seq,vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    return loss1
end

"""
forw(r::Net,x::Vector) for sequences.
x can be a Vector of Arrays representing items.
x can be a Vector of Tuples representing multiple inputs.
x cannot be a Vector of scalars (TODO:think this over)
"""
function forw(r::Net, x::Vector; yout=nothing, ygold=nothing, a...)
    # display((:forwseq0,length(x),vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
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
    # display((:forwseq1,length(x),vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
    return loss
end

# Unfortunately this should be op specific

        # elseif in(nothing, x)
        #     r.out[n] = forwnothing(r.op[n], x..., r.out0[n])

# """
# forwnothing: treat nothing as identity element, i.e. if one input is
# nothing and the other has the same size as the output return that one.
# """
# function forwnothing(op::Op, x1, x2, out0)
#     x1==x2==nothing && return nothing
#     x = (x1 == nothing ? x2 :
#          x2 == nothing ? x1 : error())
#     size(x) != size(out0) && return nothing
#     x === out0 ? out0 : copy!(out0, x)
# end

# forwnothing(op::Op, ::Void, out0)=nothing
