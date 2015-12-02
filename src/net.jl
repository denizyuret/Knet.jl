"DataType for program instructions."
immutable Ins
    output::Symbol
    op::Op
    inputs::Vector{Symbol}
    cond::Expr
    plist::Dict
end

"DataType for program registers."
type Reg
    out; out0; dif; dif0; tmp;
    size #::Dims prevents us from using nothing
    eltype::DataType
    outtype::DataType
    diftype::DataType
    tmptype::DataType
    plist::Dict
    Reg()=(r=new();r.plist=Dict();r)
end

"DataType for a compiled network."
type Net <: Model
    prog::Vector{Ins}
    reg::Dict{Symbol,Reg}
    stack::Vector
    sp::Int
    lastforw
end

function Net(prog::Vector{Ins})
    reg = [ x.output=>Reg() for x in prog ]
    Net(prog, reg, Any[], 0, nothing)
end



### DEAD CODE

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
#     prog = _knet(b)
#     net.op = map(first, prog)
#     N = length(prog)
#     dict = Dict{Symbol,Int}()
#     for i=1:N; dict[last(prog[i])] = i; end
#     net.inputs = [ Int[] for i=1:N ]
#     for i=1:N; for j=2:length(prog[i])-1; push!(net.inputs[i], dict[prog[i][j]]); end; end
#     net.outputs = outputs(net.inputs)
#     net.netinputs = count(x->isa(x,Input), net.op)
#     net.params = filter(x->isa(x,Par), net.op)
#     net.tosave = tosave(net.op, net.inputs)     # TODO: does this not depend on dx as well?
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

