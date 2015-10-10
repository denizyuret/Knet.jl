"""
forw(r::Net,x...) executes forward pass for one item (that may or may
not be part of a sequence), and returns the output.  The input x can
consist of zero or more arrays.  The arrays can be any Julia array
type (gpu/cpu, dense/sparse, Float64/32/16), they will get copied.
The output is always a dense array (on gpu if available) that is
internal to the net, and will be overwritten in the next call, so it
should be copied by the caller if needed long term.  initforw
allocates out0 if necessary but does not initialize out or out0, so
for rnn's that have forward references a call to reset at the
beginning of a sequence is necessary.
"""
function forw(r::Net, x...; trn=false, seq=false, a...)
    initforw(r, x...; a...)
    N = nops(r)
    lastinput = 0
    for n = 1:N
        trn && seq && r.tosave[n] && push(r,n)
        if isa(r.op[n], Input)
            r.out[n] = copy!(r.out0[n], x[lastinput += 1])
        else
            xn = r.out[r.inputs[n]]  # t:13/628
            r.out[n] = forw(r.op[n], xn..., r.out0[n]; trn=trn, a...)
        end
    end
    return r.out[N]
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

