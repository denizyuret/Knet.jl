"""
initforw allocates register storage (out0) and sets register fields
and flags if these are uninitialized or there is a change in input
size, type, or keyword arguments.  It does not zero or reset the
registers, there is a separate reset!  function for that.  FNN's do
not need reset, and RNNs only do at the beginning of a sequence.  Here
RNN refers to any network with a read-before-write register regardless
of whether it is applied to a sequence.
"""
function initforw(f::Net, inputs...; o...)
    nextforw = (map(typeof, inputs), map(size, inputs), o)
    if isvoid(f,:lastforw) || f.lastforw!=nextforw
        #setp(f,:forwoverwrite,false); setp(f,:backoverwrite,false)
        if isvoid(f,:lastforw) || (o != f.lastforw[3]) # when conditions change, the comp graph changes, effects argv, fanout, save
            initcond(f; o...)       # o,name,cond -> :forw
            initargv(f)             # :forw,name,args -> argv
            initsave(f)             # :forw,op,argv -> :save
            initfanout(f)           # :forw,name,argv -> :fanout
        end                         # changing input size/type or argv may change size and type of registers
        initsize(f, inputs...)      # :forw,inputs,op -> :size
        inittype(f, inputs...)      # :forw,inputs,op -> :eltype,:outtype
        initout0(f)                 # :forw,argv,op,:outtype,:eltype,:size,:forwoverwrite,:save,:fanout -> out0
        f.lastforw = nextforw
        gpusync()
    end
    length(inputs) == ninputs(f) || throw(ArgumentError())
end

initout0(f::Net;o...)=(for p in forwregs(f); initout0(f,p;o...); end)

function initout0(f::Net, p::Reg; seq=false)
    at, et, sz = getp(p,:outtype), getp(p,:eltype), getp(p,:size)
    if checkarray(p, :out0, at, et, sz)
        # all done
    elseif !isvoid(p, :out0) && (isa(p.op, Par) || isa(p.op, Arr))
        error("Size or type change not allowed in parameters and constants")
    else
        p.out0 = nothing
        if canoverwrite(p.op) && getp(p, :forwoverwrite, true)
            for i in reverse(p.argv) # consider overwriting the last input first
                q = reg(f,i)
                if canshare(f,p,q)
                    p.out0 = q.out0
                    @dbg info("Overwrite $(findfirst(f.reg,q))->$(findfirst(f.reg,p))=$(Int(pointer(p.out0))%1000)")
                    break
                end
            end
        end
        if p.out0 == nothing
            p.out0 = newarray(at, et, sz)
            gpusync()
            @dbg info("Alloc $(findfirst(f.reg,p))=$(Int(pointer(p.out0))%1000)")
        end
    end
end

function checkarray(r::Reg, n::Symbol, atype::DataType, etype::DataType, dims::Dims)
    isdefined(r,n) &&
    isa(r.(n), atype) &&
    eltype(r.(n)) == etype &&
    size(r.(n)) == dims
end

function canshare(f::Net, p::Reg, q::Reg)
    # checkarray(q,:out0,at,et,sz) && !ispersistent(q) && !getp(q, :save) && getp(q, :fanout)==1
    ispersistent(q) && return false
    getp(q,:save) && return false
    getp(q,:fanout) > 1 && return false
    at, et, sz = getp(p,:outtype), getp(p,:eltype), getp(p,:size)
    checkarray(q,:out0,at,et,sz) || return false
    # a=b*c; c=sigm(a) should not share, dot cannot overwrite!
    # a=b*c; d=e*f; e=a+d; e should not share a, a overwrites it!
    p.out0 = q.out0
    ok = checkshare(f)
    p.out0 = nothing
    return ok
end

isvoid(x,n)=(!isdefined(x,n) || isa(x.(n),Void))

function checkshare(f::Net)
    d = ObjectIdDict()
    for i=1:length(f)
        ri=f.reg[i]
        getp(ri,:forw) || continue
        !isvoid(ri,:out0) || continue
        d[ri.out0]=i
    end
    for i=1:length(f)
        ri=f.reg[i]
        getp(ri,:forw) || continue
        for j in ri.argv
            rj=f.reg[j]
            !isvoid(rj,:out0) || continue
            k = d[rj.out0]
            k == j || return false # warn("$j was overwritten by $k before being read by $i.")
            !isvoid(ri,:out0) || continue
            !canoverwrite(ri.op) && ri.out0===rj.out0 && return false # warn("$i and $j share array in nonoverwriting op.")
        end
        !isvoid(ri,:out0) && (d[ri.out0] = i)
    end
    return true
end

newarray(::Type{Array}, t::DataType, d::Dims)=Array(t,d)
newarray(::Type{SparseMatrixCSC}, t::DataType, d::NTuple{2,Int})=SparseMatrixCSC(d[1], d[2], ones(Cint, d[2]+1), Array(Cint, 0), Array(t, 0))
@gpu newarray(::Type{CudaArray}, t::DataType, d::Dims)=CudaArray(t,d)
@gpu newarray(::Type{CudaSparseMatrixCSC}, t::DataType, d::NTuple{2,Int})=CudaSparseMatrixCSC(t, fillsync!(CudaArray(Cint, d[2]+1), 1), CudaArray(Cint, 0), CudaArray(t, 0), d)
@gpu newarray(::Type{CudaSparseMatrixCSR}, t::DataType, d::NTuple{2,Int})=CudaSparseMatrixCSR(t, fillsync!(CudaArray(Cint, d[1]+1), 1), CudaArray(Cint, 0), CudaArray(t, 0), d)
@gpu newarray(::Type{CudaSparseMatrixCSCU}, t::DataType, d::NTuple{2,Int})=CudaSparseMatrixCSCU(t, d...)
@gpu newarray(::Type{CudaSparseMatrixCSRU}, t::DataType, d::NTuple{2,Int})=CudaSparseMatrixCSRU(t, d...)

# We do not need to copy persistent registers, they are guaranteed not to change during forwback.
ispersistent(p::Reg)=(isa(p.op,Par) || isa(p.op,Arr))
isreturn(p::Reg)=(p.name==:return)


# initcond(f::Net; o...) sets the :forw property of each f.reg[n] which
# is determined by evaluating f.reg[n].cond in the context of the
# kwargs passed to forw.

function initcond(f::Net; o...)
    for p in regs(f)
        setp(p,:forw,false)
    end
    d = Dict(o)
    for p in regs(f)
        c = condeval(p.cond,d)
        setp(p,:forw,c)
        c && isreturn(p) && break
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

function forwregs(f::Net)
    a = Reg[]
    for r in regs(f)
        getp(r,:forw) && push!(a,r)
    end
    return a
end

function initargv(f::Net)
    # There could be repeated and read-before-write variables
    s2i = Dict{Symbol,Int}()
    for n=1:length(f)
        p = reg(f,n)
        getp(p,:forw) || continue
        s2i[p.name] = n
    end
    for n=1:length(f)
        p = reg(f,n)
        getp(p,:forw) || continue
        p.argv = convert(Vector{Int}, map(s->s2i[s], p.args))
        s2i[p.name] = n
    end
end

function initsize(f::Net, inputs...)
    for p in forwregs(f)
        setp(p,:size,nothing)
    end
    lastinput = 0
    for p in forwregs(f)
        isa(p.op, Input) || continue
        setp(p,:size,size(inputs[lastinput += 1]))
    end
    nodims(p)=(s=getp(p,:size); s==nothing || prod(s)==0)
    notfound = count(nodims, forwregs(f))
    while notfound > 0
        for p in forwregs(f)
            isa(p.op, Input) && continue
            dy = getp(p,:size)
            xx = inputregs(f,p)
            dx = map(x->getp(x,:size), xx)
            d = infersize(p.op, dx..., dy)
            d == nothing && continue
            setp(p,:size,d[end])
            for i=1:length(xx)
                setp(xx[i],:size,d[i])
            end
        end
        nf = count(nodims, forwregs(f))
        nf == notfound && error("Cannot infer sizes")
        notfound = nf
    end
end

# inittype: Determines element and array type of p.out0 for each register p.
function inittype(f::Net, inputs...)
    isempty(inputs) && error("Don't know how to infer eltype with inputless networks yet.")
    ftype = eltype(inputs[1])
    for i in inputs
        eltype(i) == ftype || error("Conflicting element type in input $i.")
    end
    lastinput = 0
    for p in forwregs(f)
        setp(p,:eltype,ftype)
        if isa(p.op,Input) && issparse(inputs[lastinput+=1])
            setp(p,:outtype, (gpu() ? CudaSparseMatrixCSC : SparseMatrixCSC))
        else
            setp(p,:outtype, (gpu() ? CudaArray : Array))
        end
    end
end

# initsave(f::Net) sets the :save property of each f.prog[n] which
# should be true if the result of f.prog[n] will be needed for back
# calculation.  We find this out using back_reads_x and back_reads_y
# on each op as well as checking if the output will be returned and
# thus will be needed for back loss and grad calculation.  Any
# register with a save flag should not be overwritten in forw.  We'll
# be conservative and also not overwrite in sforw, even though it
# saves registers on the stack.

function initsave(f::Net)
    for p in regs(f); setp(p,:save,false); end
    for p in forwregs(f)
        if isreturn(p) || back_reads_y(p.op)
            setp(p,:save,true)
        end
        if back_reads_x(p.op)
            for i in p.argv
                setp(reg(f,i),:save,true)
            end
        end
    end
end

# getp(p,:fanout)>1: these registers are read at multiple locations in
# the program.  Depends on :cond.  :incr flag set for both sback and
# back.  The return registers are read by the user and compared to
# gold, which should count as 1.  All Par registers become
# multi-output during sequences and need :incr in sback.  Multi-output
# registers can forwoverwrite but should not be overwritten.  More
# specifically they could be overwritten by the last reader, but that
# is probably not worth the complexity.

function initfanout(f::Net)
    regs = forwregs(f)
    for p in regs; setp(p,:fanout,0); end
    for p in regs
        isreturn(p) && incp(p,:fanout)
        for i in inputregs(f,p)
            @assert getp(i,:forw)
            incp(i,:fanout)
        end
    end
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
    #     fillsync!(r.out, nothing)
    # elseif keepstate
    #     copysync!(r.out, r.out0)
    # else
    #     # @assert all((r.out .== nothing) | (r.out .== r.out0))
    #     fillsync!(r.out, nothing)
    # end

# keepstate=false, seq=false

# Switching to user controlled register sharing instead of automatic optimization...
# Safer.  Also makes reg(f,:name) work as expected.

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
#     sizes = infersize(f, inputs...) # todo: fix infersize
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
#     # todo: infersize should make sure same register has same size
#     # should we leave array sharing to the user? x=relu(x)
#     # what if they type x=dot(x,y)?
#     # how about array sharing going back?  different rules?
#     for n=1:nops(f)
#         p = f.prog[n]
#         # todo: this is all wrong, the register has out not  p.
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
#     # todo: figure out tmp
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
# # todo: deal with inputless networks:

# ispersistent(p::Ins)=(isa(p.op,Par) || isa(p.op,Arr) || getp(p,:push))


# initsize infers the size of each register

        # This will happen with conditionals:
        # save && f.lastforw[1] && Base.warn_once("Net reset during RNN training")
        # initsave(f, save)       # set :push
