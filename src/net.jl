"DataType for program registers."
type Reg
    op::Op
    name::Symbol
    args::Vector{Symbol}
    cond::Expr
    argv::Vector{Int}
    plist::Dict{Symbol,Any}
    out; out0; dif; dif0; tmp;
    Reg(op::Op,name::Symbol,args::Vector{Symbol},cond::Expr)=new(op,name,args,cond,Int[],Dict{Symbol,Any}())
end

"DataType for a compiled network."
type Net <: Model
    reg::Vector{Reg}
    stack::Vector
    sdict::ObjectIdDict         # To implement copy-on-write
    lastforw
    lastback
    Net(reg::Vector{Reg})=new(reg, Any[], ObjectIdDict())
end

import Base: length, get, eltype, pop!, push!

registers(f::Net)=f.reg
length(f::Net)=length(f.reg)
get(f::Net,i)=f.reg[i]          # i could be an array or any other type of index expression
params(f::Net)=filter(x->isa(x,Par),map(x->x.op,f.reg))
ninputs(f::Net)=count(x->isa(x.op,Input),f.reg)
eltype(f::Net)=(r=f.reg[1];isdefined(r,:out0)?eltype(r.out0):error("Uninitialized Net"))

function get(f::Net,k::Symbol)
    i = findlast(f.reg) do p
        get(p,:forw) && p.name==k
    end
    return i==0 ? nothing : f.reg[i]
end

out(f::Net,k::Symbol)=(r=get(f,k); r==nothing ? r : r.out)
dif(f::Net,k::Symbol)=(r=get(f,k); r==nothing ? r : r.dif)

# map too slow?
# inputs(f::Net,p::Reg)=map(x->x.out, input_registers(f,p))
function inputs(f::Net,p::Reg)
    n = length(p.argv)
    a = cell(n)
    @inbounds for i=1:n; a[i]=f.reg[p.argv[i]].out; end
    return a
end

input_registers(f::Net,p::Reg)=f.reg[p.argv]

get(p::Reg,k::Symbol,v=false)=get(p.plist,k,v)
set!(p::Reg,k::Symbol,v=true)=(p.plist[k]=v)
inc!(p::Reg,k::Symbol)=set!(p,k,1+get(p,k,0))
set!(f::Net,k::Symbol,v=true)=(for p in registers(f); p.plist[k]=v; end)

function push!(f::Net,a)
    push!(f.stack,a)
    if a!=nothing
        isa(a,NTuple{3}) || error("Expected NTuple{3} got $a")
        (y, xsave, ysave) = a
        ysave == nothing || inc!(f.sdict,ysave)
        xsave == nothing || (for x in xsave; inc!(f.sdict,x); end)
    end
end

pop!(f::Net)=pop!(f.stack)

inc!(p::ObjectIdDict,k)=(p[k]=true)

# Too expensive:

# function pop!(f::Net)
#     a = pop!(f.stack)
#     if a!=nothing
#         isa(a,NTuple{3}) || error("Expected NTuple{3} got $a")
#         (y, xsave, ysave) = a
#         ysave == nothing || dec!(f.sdict,ysave)
#         xsave == nothing || (for x in xsave; dec!(f.sdict,x); end)
#     end
#     return a
# end

# inc!(p::ObjectIdDict,k)=(p[k]=1+get(p,k,0))

# function dec!(p::ObjectIdDict,k)
#     haskey(p,k) || (warn("Object not in dict"); return)
#     n = get(p,k,0)
#     if n > 1
#         p[k] = n-1
#     elseif n == 1
#         delete!(p,k)
#     else
#         warn("Bad count for object: $n")
#         delete!(p,k)
#     end
# end

### Cleanup at the beginning/end of sequence

function reset!(f::Net; keepstate=false, a...) # TODO: get rid of keepstate, rnnlm defined its own reset
    isempty(f.stack) || warn("Stack not empty")
    isempty(f.sdict) || warn("Sdict not empty")
    empty!(f.stack)
    empty!(f.sdict)
    for p in registers(f)
        p.out = (keepstate ? p.out0 : nothing)
        p.dif = nothing
        get(p,:incr) && fillsync!(p.dif0, 0)
    end
    # Base.show_backtrace(STDOUT,backtrace());println()
end

### DEAD CODE

# op(f::Net,n::Int)=f.reg[n].op
# inputs(f::Net,n::Int)=f.reg[n].inputs
# output(f::Net,n::Int)=f.reg[n].output
# forwref(f::Net,n::Int)=any(i->in(output(f,n),inputs(f,i)), 1:n-1)


# registers(f::Net)=values(f.reg)
# output_register(f::Net,p::Ins)=get(f.reg,p.output,nothing)
# input_registers(f::Net,p::Ins)=map(s->get(f.reg,s,nothing), p.inputs)
# input_arrays(f::Net,p::Ins)=map(s->(haskey(f.reg,s) ? f.reg[s].out : nothing), p.inputs)

# nops(f::Net)=length(f.reg)

# getprop(p::Ins,k,d=false)=get(p.plist,k,d)
# setprop!(p::Ins,k,v=true)=(p.plist[k]=v)
# set!(p::Ins,k,v=true)=setprop!(p,k,v)

# getprop(p::Reg,k,d=false)=get(p.plist,k,d)
# setprop!(p::Reg,k,v=true)=(p.plist[k]=v)
# set!(p::Reg,k,v=true)=setprop!(p,k,v)
# inc!(p::Reg,k)=set!(p,k,1+get(p,k))

# getreg(f::Net,k::Symbol)=get(f.reg,k,nothing)
# getdif(f::Net,k::Symbol)=(haskey(f.reg,k) ? f.reg[k].dif : nothing)
# getout(f::Net,k::Symbol)=(haskey(f.reg,k) ? f.reg[k].out : nothing)
# getreg(f::Net,p::Ins)=getreg(f,p.output)
# getdif(f::Net,p::Ins)=getdif(f,p.output)
# getout(f::Net,p::Ins)=getout(f,p.output)

# Base.get(f::Net,k)=getout(f,k)
# Base.get(p::Ins,k,d=false)=getprop(p,k,d)
# Base.get(p::Reg,k,d=false)=getprop(p,k,d)

# Base.copysync!(r::Reg,x)=(r.out=copysync!(r.out0,x))

# type Ins
#     output::Symbol
#     op::Op
#     inputs::Vector{Symbol}
#     cond::Expr
#     plist::Dict
#     out; out0; dif; dif0; tmp
#     function Ins(output::Symbol,op::Op,inputs::Vector{Symbol},cond::Expr)
#         new(output,op,inputs,cond,Dict(),nothing,nothing,nothing,nothing,nothing)
#     end
# end

# type Net
#     op::Vector{Ins}
#     sp::Int
#     stack::Vector
# end


# """
# tosave(op, inputs) returns tosave[n] which is true if the result of 
# op[n] would be needed for back calculation.  We find this out using 
# back_reads_x and back_reads_y on each op.  Note that Par registers 
# are persistent and do not need to be saved.
# """
# function tosave(op, inputs)
#     N = length(op)
#     tosave = falses(N)
#     for n=1:length(op)
#         back_reads_y(op[n]) && (tosave[n] = true)
#         if back_reads_x(op[n])
#             for i in inputs[n]
#                 if !isa(op[i], Par)
#                     tosave[i] = true
#                 end
#             end
#         end
#     end
#     return tosave
# end

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

# """
# Net(::Expr) compiles a quoted block of net language to a Net object
# """
# function Net(a::Expr)
#     (op, inputs) = netcomp(a)
#     N = length(op)
#     Net(op, inputs, outputs(inputs),
#         count(x->isa(x,Input), op),
#         filter(x->isa(x,Par), op),
#         tosave(op, inputs),
#         falses(N), # toback: depends on dx
#         falses(N), # toincr: depends on seq
#         falses(N), # sparse: depends on input
#         cell(N), cell(N), cell(N), cell(N), cell(N),
#         Any[], 0)
# end

# """
# netcomp(::Expr) compiles Expr and returns (ops, opinputs)
# where inputs are represented as integer indices.
# """
# function netcomp(a::Expr)
#     (op,innames,outname) = netcomp1(a)
#     dict = Dict{Symbol,Int}()
#     for n=1:length(outname)
#         dict[outname[n]] = n
#     end
#     inputidx = Array(Vector{Int}, length(innames))
#     for n=1:length(innames)
#         inputidx[n] = map(x->dict[x], innames[n])
#     end
#     (op, inputidx)
# end

# """
# netcomp1(::Expr) compiles Expr and returns (ops, inputs, output)
# where inputs and outputs are gensyms
# """
# function netcomp1(block::Expr)
#     @assert block.head == :block
#     dict = Dict{Symbol,Symbol}()
#     ops = Op[]
#     inputs = Vector{Symbol}[]
#     output = Symbol[]
#     for stmt in block.args
#         isa(stmt, LineNumberNode) && continue
#         @assert isa(stmt, Expr)
#         (opoutput, func, opinputs, params) = netstmt(stmt, dict)
#         ev = eval(current_module(), Expr(:call, func, params...))
#         if isa(ev, Op)
#             @assert length(opinputs) == ninputs(ev)
#             push!(ops, ev)
#             push!(inputs, opinputs)
#             push!(output, opoutput)
#         elseif isa(ev, Expr)
#             (subops, subinputs, suboutput) = subcomp(ev, opinputs, opoutput)
#             append!(ops, subops)
#             append!(inputs, subinputs)
#             append!(output, suboutput)
#         else
#             error("Compiler error: $ev, $func, $params")
#         end
#     end
#     (ops, inputs, output)
# end

# """
# subcomp(::Expr, netinputs, netoutput): compiles a subroutine with 
# input/output symbols given by the caller.
# """
# function subcomp(block::Expr, netinputs, netoutput) 
#     (ops, inputs, output) = netcomp1(block)
#     @assert length(netinputs) == count(op->isa(op,Input), ops)
#     substitute = Dict{Symbol,Symbol}()
#     nextinput = 0
#     for i=1:length(ops)
#         isa(ops[i], Input) || continue 
#         substitute[output[i]] = netinputs[nextinput += 1]
#     end
#     substitute[output[end]] = netoutput
#     newops = Any[]
#     newinputs = Any[]
#     newoutput = Any[]
#     for i=1:length(ops)
#         isa(ops[i], Input) && continue                          # input ops can be dropped, inputs are now caller symbols
#         push!(newops, ops[i])
#         push!(newinputs, map(x->get(substitute,x,x), inputs[i]))
#         push!(newoutput, get(substitute, output[i], output[i]))
#     end
#     (newops, newinputs, newoutput)
# end

# """
# netstmt(::Expr,::Dict): parses a single assignment statement converting 
# variables to gensyms.  Returns (target, func, args, pars).
# """
# function netstmt(stmt::Expr, dict::Dict{Symbol,Symbol})
#     @assert stmt.head == :(=)
#     @assert length(stmt.args) == 2
#     (target, expr) = stmt.args
#     @assert isa(target, Symbol)
#     target = get!(dict, target) do 
#         gensym(target) 
#     end
#     @assert expr.head == :call
#     func = expr.args[1]
#     args = Any[]
#     pars = Any[]
#     for a in expr.args[2:end]
#         isa(a, Symbol) ?
#         push!(args, a) :
#         push!(pars, a)
#     end
#     for i=1:length(args)
#         args[i] = get!(dict, args[i]) do
#             gensym(args[i]) 
#         end
#     end
#     (target, func, args, pars)
# end


# """
# multi(op, inputs) returns a bool vector which is true if op[n] has 
# fanout > 1, in which case its dif should be incrementally updated.
# """
# function multi(op, inputs)
#     N = length(op)
#     nout = zeros(Int, N)
#     nout[N] = 1  # count network output as a read
#     for n=1:N
#         for i in inputs[n]
#             nout[i] += 1
#         end
#     end
#     return (nout .> 1)
# end

# """
# Neural network compiler.
# """
# type Net0
#     op::Vector{Op}
#     inputs::Vector{Vector{Int}}
#     outputs::Vector{Vector{Int}}
#     netinputs::Int
#     params::Vector{Par}
#     tosave::Vector{Bool}        # result of op[n] needed for back calculation.  
#     toback::Vector{Bool}        # dif[n] should be calculated during back calculation
#     toincr::Vector{Bool}        # dif[n] should be incrementally updated: multiple outputs or Par in a sequence model
#     sparse::Vector              # a parameter dotted with sparse input
#     out::Vector
#     dif::Vector
#     out0::Vector
#     dif0::Vector
#     tmp::Vector
#     stack::Vector
#     sp::Int
#     Net0()=new()
# end
    
# function Net(f::Function; ninputs=1, o...)
#     x = [ gensym("x") for i=1:ninputs ]
#     y = gensym("y")
#     p = f(x..., y; o...)
#     b = Expr(:block)
#     for i in x; push!(b.args, :($i = input())); end
#     append!(b.args, p.args)
#     Net(b)
# end

# function Net(b::Expr)
#     net = Net()
#     reg = _knet(b)
#     net.op = map(first, reg)
#     N = length(reg)
#     dict = Dict{Symbol,Int}()
#     for i=1:N; dict[last(reg[i])] = i; end
#     net.inputs = [ Int[] for i=1:N ]
#     for i=1:N; for j=2:length(reg[i])-1; push!(net.inputs[i], dict[reg[i][j]]); end; end
#     net.outputs = outputs(net.inputs)
#     net.netinputs = count(x->isa(x,Input), net.op)
#     net.params = filter(x->isa(x,Par), net.op)
#     net.tosave = tosave(net.op, net.inputs)     # todo: does this not depend on dx as well?
#     net.toback = falses(N) # toback: depends on dx
#     net.toincr = falses(N) # toincr: depends on seq
#     net.sparse = nothings(N) # sparse: depends on input
#     for f in (:out,:dif,:out0,:dif0,:tmp); net.(f) = nothings(N); end
#     net.stack = Any[]
#     net.sp = 0
#     return net
# end

# No longer true, we are switching to user controlled register based sharing:
# Two registers may share their out0 arrays but not dif0 or vice-versa
# So each symbol should correspond to a unique register, sharing
# optimization can be done at the array level, not the register level.

# "DataType for regram registers."
# type Reg
#     size #::Dims prevents us from using nothing during size inference, TODO; use empty tuple instead, TODO: should we put these into plist?
#     eltype::DataType
#     outtype::DataType
#     diftype::DataType
#     tmptype::DataType
#     Reg()=(r=new();r.plist=Dict();r.saved=false;r)
# end

