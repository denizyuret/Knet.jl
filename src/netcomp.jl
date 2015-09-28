"""
Neural network.
"""
immutable Net <: Model
    op::Vector{Op}
    inputs::Vector{Vector{Int}}
    netinputs::Int
    params::Vector{Par}
    multi::Vector{Bool}
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
    Net(op, inputs,
        count(x->isa(x,Input), op),
        filter(x->isa(x,Par), op),
        multi(op, inputs),
        tosave(op, inputs),
        toback(op, inputs),
        falses(N),
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
tosave(op, inputs) returns a Bool vector which is true if the result of 
op[n] should be saved for back calculation.
"""
function tosave(op, inputs)
    N = length(op)
    tosave = falses(N)
    for n=1:N
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
multi(op, inputs) returns a bool vector which is true if op[n] has 
fanout > 1, in which case its dif should be incrementally updated.
"""
function multi(op, inputs)
    N = length(op)
    nout = zeros(Int, N)
    nout[N] = 1  # count network output as a read
    for n=1:N
        for i in inputs[n]
            nout[i] += 1
        end
    end
    return (nout .> 1)
end

"""
toback(op, inputs) returns a boolean vector which is true if dif[n] should be
calculated for op[n] during back calculation.  This is only needed if 
op[n] is a par node or a par node descendent.
"""
function toback(op, inputs)
    N = length(op)
    toback = falses(N)
    for n=1:N
        isa(op[n], Par) && (toback[n] = true)
    end
    nback = sum(toback)
    while true
        for n=1:N
            toback[n] && continue
            for i in inputs[n]
                toback[i] || continue
                toback[n] = true
                break
            end
        end
        nb = sum(toback)
        nb == nback ? break : nback = nb
    end
    return toback
end

# DEPRECATED: going back to initializing with nothings
# """
# tozero(op, inputs) returns a boolean vector which is true if out0[n] should be
# zeroed out before the forw calculation.  This is only necessary if it is read
# before it is written.
# """
# function tozero(op, inputs)
#     N = length(op)
#     tozero = falses(N)
#     for n=1:N
#         for i in inputs[n]
#             if i > n
#                 tozero[i] = true
#             end
#         end
#     end
#     return tozero
# end
