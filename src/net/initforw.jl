"""
initforw allocates register storage (out0) and sets the :forw and
:push flags if these are uninitialized or there is a change in input
size, sparsity, save flag, or keyword arguments.  It does not zero or
reset the registers, there is a separate reset!  function for that.
FNN's do not need reset, and RNNs only do at the end of a sequence.
"""
function initforw(f::Net, inputs...; save=false, o...)
    @assert length(inputs) == ninputs(f)
    if (f.lastforw == nothing
        || save != f.lastforw[1]
        || map(size, inputs) != f.lastforw[2]
        || map(stype, inputs) != f.lastforw[3]
        || sort(o) != f.lastforw[4])
        save && f.lastforw[1] && Base.warn_once("Net reset during RNN training")
        initcond(f; o...)       # set :forw
        initsave(f, save)       # set :push
        initsize(f, inputs...)  # set reg.size
        inittype(f, inputs...)  # set reg.eltype, reg.outtype
        initout0(f, inputs...)  # set reg.out0
        f.lastforw = (save, map(size, inputs), map(stype, inputs), sort(o))
    end
end


# initcond(f::Net; o...) sets the :forw property of each f.prog[n] which
# is determined by evaluating f.prog[n].cond in the context of the
# kwargs passed to forw.

function initcond(f::Net; o...)
    for p in f.prog; setprop!(p,:forw,false); end
    d = Dict(o)
    for p in f.prog
        c = condeval(p.cond,d)
        setprop!(p,:forw,c)
        c && (p.output == :return) && break
    end
end

function condeval(s,d::Dict)
    (isa(s, Bool) ? s :
     isa(s, Symbol) ? get(d,s,false) :
     !isa(s, Expr) ? error("Expected boolean expression got $s") :
     s.head == :&& ? mapreduce(x->condeval(x,d), &, true, s.args) :
     s.head == :|| ? mapreduce(x->condeval(x,d), |, false, s.args) :
     s.head == :call && s.args[1] == :! ? !condeval(s.args[2],d) :
     error("Expected boolean expression got $s"))
end

# initsave(f::Net) sets the :push property of each f.prog[n] which
# should be true if the result of f.prog[n] will be needed for back
# calculation.  We find this out using back_reads_x and back_reads_y
# on each op.  Note that Par registers are persistent and do not need
# to be saved.  Also note that save is only necessary for (1) RNN
# sequence models and (2) only for training and (3) only for ops with
# forw=true.

function initsave(f::Net, save::Bool)
    for p in f.prog; setprop!(p,:push,false); end
    save || return
    for p in f.prog
        get(p,:forw) || return
        back_reads_y(p.op) && setprop!(p,:push,true)
        if back_reads_x(p.op)
            for q in f.prog
                if in(q.output, p.inputs) && !ispersistent(q)
                    setprop!(q,:push,true)
                end
            end
        end
    end
end

ispersistent(p::Ins)=(isa(p.op,Par) || isa(p.op,Arr) || get(p,:push))


# initsize infers the size of each register
# TODO: ignore the ones with forw=false?

function initsize(f::Net, inputs...)
    for (n,r) in f.reg; r.size = nothing; end
    lastinput = 0
    for p in f.prog
        isa(p.op, Input) || continue
        r = f.reg[p.output]
        s = size(inputs[lastinput += 1])
        r.size == nothing ? (r.size = s) : (@assert r.size==s)
    end
    nodims(x)=(x.size==nothing || prod(x.size)==0)
    notfound = count(nodims, values(f.reg))
    while notfound > 0
        for p in f.prog
            isa(p.op, Input) && continue
            y = f.reg[p.output]
            dy = y.size
            dx = [ f.reg[i].size for i in p.inputs]
            d = infersize(p.op, dx..., dy)
            d == nothing && continue
            y.size = d[end]
            for i=1:length(p.inputs)
                f.reg[p.inputs[i]].size = d[i]
            end
        end
        nf = count(nodims, values(f.reg))
        nf == notfound && error("Cannot infer sizes")
        notfound = nf
    end
end

# Determine element and array type of r.out0 for each register r.
function inittype(f::Net, inputs...)
    isempty(inputs) && error("Don't know how to infer eltype with inputless networks yet.")
    ftype = eltype(inputs[1])
    for i in inputs
        eltype(i) == ftype || error("Conflicting element type in input $i.")
    end
    lastinput = 0
    for p in f.prog
        r = f.reg[p.output]
        r.eltype = ftype        # TODO: What if there is a conflict with par.init or a previous eltype?
        if isa(p.op,Input) && issparse(inputs[lastinput+=1])
            r.outtype = (gpu() ? CudaSparseMatrixCSC : SparseMatrixCSC)
        else                    # TODO: not all these types will allow r.outtype(r.eltype,r.size)
            r.outtype = (gpu() ? CudaArray : Array)
        end
    end
end

function initout0(f::Net, inputs...)
    for p in f.prog
        get(p,:forw) || continue
        r = f.reg[p.output]
        if checkarray(r, :out0, r.outtype, r.eltype, r.size)
            # all done
        elseif isdefined(r,:out0) && (isa(p.op, Par) || isa(p.op, Arr))
            error("Size or type change not allowed in parameters")
        else
            r.out0 = r.outtype(r.eltype, r.size)
        end
    end
end

function checkarray(r::Reg, n::Symbol, atype::DataType, etype::DataType, dims::Dims)
    isdefined(r,n) &&
    isa(r.(n), atype) &&
    eltype(r.(n)) == etype &&
    size(r.(n)) == dims
end

### DEAD CODE:


# """
# outputs(inputs) returns an array of output indices for each register.
# Note that the final register is the network output, so it could be 
# read externally even if outputs[N] is empty.
# """
# function outputs(inputs)
#     N = length(inputs)
#     outputs = [ Int[] for n=1:N ]
#     for n=1:N
#         for i in inputs[n]
#             push!(outputs[i], n)
#         end
#     end
#     push!(outputs[N], 0)        # for network output
#     return outputs
# end


    # if ygold != nothing
    #     @assert issimilar2(ygold, r.out0[N])
    #     if (r.dif0[ N]!=nothing)
    #         @assert issimilar3(ygold, r.dif0[N])
    #     else
    #         r.dif0[N] = newarray(gpu(), stype(ygold), eltype(ygold), size(ygold))
    #     end
    # end


    # if !seq
    #     # @assert all((r.out .== nothing) | (r.out .== r.out0)) # t:110/244
    #     @assert !keepstate "meaningless keepstate in non-sequence run"
    #     fill!(r.out, nothing)
    # elseif keepstate
    #     copy!(r.out, r.out0)
    # else
    #     # @assert all((r.out .== nothing) | (r.out .== r.out0))
    #     fill!(r.out, nothing)
    # end

# keepstate=false, seq=false

# Switching to user controlled register sharing instead of automatic optimization...
# Safer.  Also makes get(f,:name) work as expected.

# function findout0(f::Net, n, sizes, nsparse)
#     qn = f.prog[n]
#     ispersistent(qn) && return
#     free = nothing               # search most recent written first to avoid copying in overwriting ops
#     for i = n-1:-1:1             # considering overwriting i with n
#         qi = f.prog[i]
#         ispersistent(qi) && continue
#         !overwrites(qn.op) && in(qi.output, qn.inputs) && continue
#         ri = f.reg[qi.output]
#         isdefined(ri,:out0) || continue
#         size(ri.out0) == sizes[n] || continue
#         stype(ri.out0) == nsparse || continue
#         willberead = false                              # is anybody going to read i before it is written again?
#         k = n
#         while true
#             k = mod1(k+1, length(f.prog))
#             qk = f.prog[k]
#             for j in qk.inputs
#                 rj = f.reg[j]
#                 isdefined(rj,:out0) && rj.out0 === ri.out0 && (willberead = true; break)
#             end
#             willberead && break
#             rk = f.reg[qk.output]
#             isdefined(rk,:out0) && rk.out0 === ri.out0 && break
#         end
#         !willberead && (free = ri.out0; break)
#     end
#     return free
# end

# function initout0(f::Net, inputs...; o...)
#     xtype = infertype(f, inputs...)
#     sizes = infersize(f, inputs...) # TODO: fix infersize
#     lastinput = 0
#     for p in f.prog
#         r = f.reg[p.output]
#         # We may keep arrays pre-allocated due to parameter sharing or batch resizing
#         # Only safe for Par/Arr, others may have sharing that no longer is valid
#         if isdefined(r,:out0) && (isa(p.op,Par) || isa(p.op,Arr))
#             size(r.out0)==sizes[n] || error("Size incompatible with network parameters")
#         else
#             r.out0 = nothing
#         end
#     end
#     # TODO: infersize should make sure same register has same size
#     # should we leave array sharing to the user? x=relu(x)
#     # what if they type x=dot(x,y)?
#     # how about array sharing going back?  different rules?
#     for n=1:nops(f)
#         p = f.prog[n]
#         # TODO: this is all wrong, the register has out not  p.
#         p.out = nothing
#         if p.out0 != nothing && (isa(p.op,Par) || isa(p.op,Arr))
#             continue
#         end
#         ### ARRAY SHARING OPTIMIZATION:
#         st = (isa(p.op, Input) ? stype(inputs[lastinput += 1]) : nothing)
#         p.out0 = findout0(f, n, sizes, st)
#         if p.out0 == nothing
#             p.out0 = newarray(gpu(), st, xtype, sizes[n])
#         end
#     end
#     # TODO: figure out tmp
# end


# it = nothing
# for i in inputs
#     t = eltype(i)
#     if it == nothing
#         it = t
#     elseif t != it
#         error("Conflicting input eltypes")
#     end
# end
# it == nothing && error("Cannot infer eltype")
# return it
# # TODO: deal with inputless networks:
