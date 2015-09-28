"""
Neural network.
"""
immutable Net <: Model
    op::Vector{Op}
    inputs::Vector{Vector{Int}}
    outputs::Vector{Vector{Int}}
    netinputs::Int
    params::Vector{Par}
    tosave::Vector{Bool} 
    toback::Vector{Bool}
    toincr::Vector{Bool}
    out::Vector
    out0::Vector
    dif::Vector
    dif0::Vector
    tmp::Vector
    stack::Vector
    sp::Int
end
    
"""
Net(::Expr) compiles a quoted block of net language to a Net object
"""
function Net(a::Expr)
    (op, inputs) = netcomp(a)
    N = length(op)
    Net(op, inputs, outputs(inputs),
        count(x->isa(x,Input), op),
        filter(x->isa(x,Par), op),
        tosave(op, inputs),
        falses(N), # toback: depends on dx
        falses(N), # toincr: depends on seq
        cell(N), cell(N), cell(N), cell(N), cell(N),
        Any[], 0)
end

"""
netcomp(::Expr) compiles Expr and returns (ops, opinputs)
where inputs are represented as integer indices.
"""
function netcomp(a::Expr)
    (op,innames,outname) = netcomp1(a)
    dict = Dict{Symbol,Int}()
    for n=1:length(outname)
        dict[outname[n]] = n
    end
    inputidx = Array(Vector{Int}, length(innames))
    for n=1:length(innames)
        inputidx[n] = map(x->dict[x], innames[n])
    end
    (op, inputidx)
end

"""
netcomp1(::Expr) compiles Expr and returns (ops, inputs, output)
where inputs and outputs are gensyms
"""
function netcomp1(block::Expr)
    @assert block.head == :block
    dict = Dict{Symbol,Symbol}()
    ops = Op[]
    inputs = Vector{Symbol}[]
    output = Symbol[]
    for stmt in block.args
        isa(stmt, LineNumberNode) && continue
        @assert isa(stmt, Expr)
        (opoutput, func, opinputs, params) = netstmt(stmt, dict)
        ev = eval(Expr(:call, func, params...))
        if isa(ev, Op)
            @assert length(opinputs) == ninputs(ev)
            push!(ops, ev)
            push!(inputs, opinputs)
            push!(output, opoutput)
        elseif isa(ev, Expr)
            (subops, subinputs, suboutput) = subcomp(ev, opinputs, opoutput)
            append!(ops, subops)
            append!(inputs, subinputs)
            append!(output, suboutput)
        else
            error("Compiler error: $ev, $func, $params")
        end
    end
    (ops, inputs, output)
end

"""
subcomp(::Expr, netinputs, netoutput): compiles a subroutine with 
input/output symbols given by the caller.
"""
function subcomp(block::Expr, netinputs, netoutput) 
    (ops, inputs, output) = netcomp1(block)
    @assert length(netinputs) == count(op->isa(op,Input), ops)
    substitute = Dict{Symbol,Symbol}()
    nextinput = 0
    for i=1:length(ops)
        isa(ops[i], Input) || continue 
        substitute[output[i]] = netinputs[nextinput += 1]
    end
    substitute[output[end]] = netoutput
    newops = Any[]
    newinputs = Any[]
    newoutput = Any[]
    for i=1:length(ops)
        isa(ops[i], Input) && continue                          # input ops can be dropped, inputs are now caller symbols
        push!(newops, ops[i])
        push!(newinputs, map(x->get(substitute,x,x), inputs[i]))
        push!(newoutput, get(substitute, output[i], output[i]))
    end
    (newops, newinputs, newoutput)
end

"""
netstmt(::Expr,::Dict): parses a single assignment statement converting 
variables to gensyms.  Returns (target, func, args, pars).
"""
function netstmt(stmt::Expr, dict::Dict{Symbol,Symbol})
    @assert stmt.head == :(=)
    @assert length(stmt.args) == 2
    (target, expr) = stmt.args
    @assert isa(target, Symbol)
    target = get!(dict, target) do 
        gensym(target) 
    end
    @assert expr.head == :call
    args = Any[]
    pars = Any[]
    for a in expr.args
        isa(a, Symbol) ?
        push!(args, a) :
        push!(pars, a)
    end
    func = shift!(args)
    for i=1:length(args)
        args[i] = get!(dict, args[i]) do
            gensym(args[i]) 
        end
    end
    (target, func, args, pars)
end

"""
tosave(op, inputs) returns tosave[n] which is true if the result of 
op[n] would be needed for back calculation.  We find this out using 
back_reads_x and back_reads_y on each op.  Note that Par registers 
are persistent and do not need to be saved.
"""
function tosave(op, inputs)
    N = length(op)
    tosave = falses(N)
    for n=1:length(op)
        back_reads_y(op[n]) && (tosave[n] = true)
        if back_reads_x(op[n])
            for i in inputs[n]
                if !isa(op[i], Par) # TODO: how about con and rnd?
                    tosave[i] = true
                end
            end
        end
    end
    return tosave
end

"""
outputs(inputs) returns an array of output indices for each register.
Note that the final register is the network output, so it could be 
read externally even if outputs[N] is empty.
"""
function outputs(inputs)
    N = length(inputs)
    outputs = [ Int[] for n=1:N ]
    for n=1:N
        for i in inputs[n]
            push!(outputs[i], n)
        end
    end
    push!(outputs[N], 0)        # for network output
    return outputs
end


### DEAD CODE

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

