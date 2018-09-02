"""
REWRITE!!
forw(f::Net, input...; kwargs...) applies the compiled knet function f
to the given input.  

If f has a return statement, its result is returned, otherwise nothing
is returned.  Following forw(), the program registers can be queried
using get(f,regname).  The special register get(f,:return) contains
the return value, if any.  Note that arrays returned by get(f,regname)
are internal to the Net and will be overwritten in the next forw call.

The input consists of zero or more arrays representing a single
minibatch.  For sequence models, the input typically corresponds to a
minibatch for a single time step.  The input arrays can be any Julia
array type (gpu/cpu, dense/sparse, Float64/32/16).  They get copied
before processing, they are never overwritten.

The keyword arguments are used as boolean condition variables during
function execution.

The internal registers are not cleared between calls.  This should not
be a problem for feed forward nets and is necessary for RNNs which
have read-before-write registers.  An explicit call to reset!(f)
clears all internal registers, which may be necessary at the beginning
of a sequence for RNNs.

forw() saves the state useful for the gradient calculation on f.stack.
This is only necessary during training where the stack will be popped
by back().  During testing apply() should be used instead, which
performs the same computation but does not touch the stack.

"""
forw(f::Net, input...; kwargs...) = _forw(f,false,input...; kwargs...)
sforw(f::Net, input...; kwargs...) = _forw(f,true,input...; kwargs...)

function _forw(f::Net, seq::Bool, input...; kwargs...)
    initforw(f, input...; kwargs...)
    lastinput = 0
    for y in regs(f)
        if getp(y,:forw)
            if isa(y.op, Input)
                y.out = copysync!(y.out0, input[lastinput += 1])
            else
                y.out = forw(y.op, inputs(f,y)..., y.out0; kwargs...)
            end
            # println(:forw, (findfirst(regs(f), y), typeof(y.op), y.argv, size(y.out0), vecnorm0(y.out)))
        end
        seq && push!(f, y)
    end
    # error(:ok)
    r = reg(f,:return)
    return (r==nothing ? r : r.out)
end


### DEAD CODE:
    # yout != nothing && copysync!(yout, r.out[N])
    # loss1 = 0.0
    # if ygold != nothing
    #     r.dif[N] = copysync!(r.dif0[N], ygold)                                      # t:6/628
    #     loss1 = loss(r.op[N], r.dif[N], r.out[N])                               # t:442/628
    # end
    # return loss1

# """
# forw(r::Net,x::Vector) for sequences.
# x can be a Vector of Arrays representing items.
# x can be a Vector of Tuples representing multiple inputs.
# x cannot be a Vector of scalars (DONE:think this over)
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

# # If y.out has been saved on stack, find new storage so we do not overwrite
# function copy_on_write(f::Net,y::Reg)
#     if !ispersistent(y) && haskey(f.sdict,y.out0)
#         y.out0 = nothing
#         initout0(f,y)
#     end
# end
