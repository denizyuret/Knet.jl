"""
forw(f::Net,input...; save=false, kwargs...) applies the compiled knet
function f to the given input.  Following the forward pass, the
program registers can be queried using get(f,:name).  get(f,:return)
is special and gives the result passed to return.  Note that arrays
returned by get(f,:name) are internal to the Net and will be
overwritten in the next forw call.

The input consists of zero or more arrays.  For sequence models, the
input typically corresponds to a single item in the sequence.  The
input arrays can be any Julia array type (gpu/cpu, dense/sparse,
Float64/32/16).  They get copied before processing, they are never
overwritten.

The internal arrays are not cleared between calls.  This should not be
a problem for feed forward nets and is necessary for RNNs.  An
explicit call to reset!(f) clears all internal arrays, which may be
necessary at the beginning of a sequence for RNNs.

The boolean keyword argument `save` tells `forw` whether to save the
arrays useful for the gradient calculation on f.stack.  This is only
necessary for RNNs and only during training.  The rest of the keyword
arguments are used as boolean condition variables during program
execution.

"""
function forw(f::Net, input...; save=false, kwargs...)
    initforw(f, input...; save=save, kwargs...)
    lastinput = 0
    for p in f.prog
        getprop(p,:forw) || continue
        r = f.reg[p.output]
        getprop(p,:push) && push(f,r)
        if isa(p.op, Input)
            r.out = copy!(r.out0, input[lastinput += 1])
        else
            xn = [ f.reg[i].out for i in p.inputs]
            r.out = forw(p.op, xn..., r.out0; kwargs...)
        end
    end
end




### DEAD CODE:
    # yout != nothing && copy!(yout, r.out[N])
    # loss1 = 0.0
    # if ygold != nothing
    #     r.dif[N] = copy!(r.dif0[N], ygold)                                      # t:6/628
    #     loss1 = loss(r.op[N], r.dif[N], r.out[N])                               # t:442/628
    # end
    # return loss1

# """
# forw(r::Net,x::Vector) for sequences.
# x can be a Vector of Arrays representing items.
# x can be a Vector of Tuples representing multiple inputs.
# x cannot be a Vector of scalars (TODO:think this over)
# """
# function forw(r::Net, x::Vector, yout=nothing; ygold=nothing, a...)
#     # display((:forwseq0,length(x),vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
#     isbits(eltype(x)) && error("forw expects a minibatch")
#     x1 = (isa(x[1], Tuple) ? x[1] : (x[1],))
#     y1 = (ygold == nothing ? nothing : ygold[1])
#     initforw(r, x1...; seq=true, ygold=y1, a...)
#     loss = 0.0
#     for i=1:length(x)
#         xi = (isa(x[i], Tuple) ? x[i] : (x[i],))
#         yi = (yout == nothing ? nothing : yout[i])
#         yg = (ygold == nothing ? nothing : ygold[i])
#         loss += forw(r, xi..., yi; seq=true, ygold=yg, a...)
#     end
#     # display((:forwseq1,length(x),vecnorm0(r.out),vecnorm0(r.stack[1:r.sp])))
#     return loss
# end

# The
# output is always a dense array (on gpu if available) that is internal
# to the net, and will be overwritten in the next call, so it should be
# copied by the caller if needed long term.  initforw allocates out0 if
# necessary but does not initialize out or out0, so for rnn's that have
# forward references a call to reset at the beginning of a sequence is
# necessary.
