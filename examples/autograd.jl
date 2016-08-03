importall Base

"""
Returns a function which computes the gradient of `fun` with respect to
positional argument number `argnum`. The returned function takes the same
arguments as `fun`, but returns the gradient instead. The function `fun`
should be scalar-valued. The gradient has the same type as the argument.
"""
function grad(fun, argnum=1)
    #TODO: @attach_name_and_doc(fun, argnum, 'Gradient')
    function gradfun(args...; kwargs...)
        backward_pass(forward_pass(fun, args, kwargs, argnum)...)
    end
    return gradfun
end

# grad: forward_pass, backward_pass

function forward_pass(fun, args, kwargs, argnum)
    tape = CalculationTape()                                    # Q: what is this for, when do we need multiple tapes?
    arg_wrt = args[argnum]
    start_node = new_node(safe_type(getval(arg_wrt)), Any[tape]) 	# Q: do we need safe_type and getval?
    args = Any[args...]                                         # Q: return grad wrt all args instead of a single one?
    args[argnum] = merge_tapes(start_node, arg_wrt)             # Q: what does merge do? arg_wrt is not a node?
    end_node = fun(args...; kwargs...)                          # TODO: add error handling.
    return start_node, end_node, tape
end

# forward_pass: Node, new_node, safe_type, getval, merge_tapes
# Q: Can we get away with a single Node type?
# Q: Why do we need multiple tapes?  Hypothesis: we build a tree, not a linear tape.
# Q: If we build a tree, how do we prune non-parameter inputs?  (they won't have node types)

# There are three interacting types:
# ReverseNode is a plain type with four slots.
# CalculationTape is an array of ReverseNodes and a "complete" flag.
# Node has a value and a Dictionary of CalculationTape => last ReverseNode in the tape

type ReverseNode; parent_grad_ops; outgrads; node_type; node_value; end # TODO: define sum_outgrads
ReverseNode(node_type, node_value) = ReverseNode([], [], node_type, node_value)
sum_outgrads(rnode::ReverseNode)=sum(rnode.outgrads)            # TODO: does this work when grads are arrays?  maybe that is the reason for the strange python syntax, using the 0'th grad as the accumulator?

# type CalculationTape; tape; complete; end       # In python subclass of list with a complete flag, TODO: figure out how to add the list part.
# CalculationTape()=CalculationTape([],false)
# push!(c::CalculationTape, item) = push!(c.tape, item)           # Q: should we just have a plain array here and handle complete some other way?

# Just use a regular array for the CalculationTape, push "nothing" to mark complete
CalculationTape()=Any[]
iscomplete(a)=(!isempty(a) && a[end]==nothing)
complete!(a)=push!(a,nothing)                                   # Q: why do we need complete?

type Node; value; tapes;                                        # field tapes is a Dict()
    function Node(value, tapes)                                 # arg tapes is an Array
        self = new(value, Dict())
        for tape in tapes
            new_rnode = ReverseNode(typeof(self), value)        # Q: do we need typeof here any more if we have a single Node type?  also why define self.Rnode in python?
            push!(tape, new_rnode)
            self.tapes[tape] = new_rnode
        end
        return self
    end
end

new_node(value, tapes)=Node(value, tapes)
safe_type(value) = isa(value, Integer) ? float(value) : value
getval(x) = isa(x, Node) ? x.value : x
merge_tapes(x,y) = x            # TODO: there is trickery in its gradient

# OK, at this point we get:
# MethodError: `sin` has no method matching sin(::Node)
# It is time to implement primitive.

"""
Wraps a function so that its gradient can be specified and its invocation
can be recorded. For examples, see the docs.
"""
type Primitive; fun; grads; zero_grads; end

# Turn methods into functions that take Primitive as first arg?
Primitive(fun) = Primitive(fun, Dict(), Set())

function gradmaker(p::Primitive, argnum, ans, args, kwargs)
    try 
        p.grads[argnum](ans, args...; kwargs...)
    catch e
        if isa(e, KeyError)
            name = p.fun.env.name
            if isempty(p.grads)
                error("Gradient of $name not yet implemented.")
            else
                error("Gradient of $name w.r.t. arg number $argnum not yet implemented.")
            end
        else
            throw(e)
        end
    end
end

defgrad(p::Primitive, gradmaker, argnum=1) = (p.grads[argnum] = gradmaker)
defgrads(p::Primitive, gradmaker, argnums) = (for argnum in argnums; defgrad(p, partial(gradmaker, argnum), argnum); end)
defgrad_is_zero(p::Primitive, argnums=(0,))= (for argnum in argnums; push!(p.zero_grads, argnum); end)

# This is where the main action is:
function call(p::Primitive, args...; kwargs...)
    argvals = Any[args...]
    ops = []
    tapes = Set()
    for (i, arg) in enumerate(args)
        if isa(arg, Node)
            argvals[i] = arg.value
            if i in p.zero_grads; continue; end              # Q: who sets zero_grads, why?
            for (tape, parent_rnode) in arg.tapes                 # Node.tapes is a dict or has pair elements
                if !iscomplete(tape)
                    push!(ops, (tape, i, parent_rnode))
                    push!(tapes, tape)
                end
            end
        end
    end
    result = p.fun(argvals...; kwargs...)
    # if isa(result, NotImplemented); return result; end          # Q: what's this for?  NotImplemented is a Python primitive!
    if !isempty(ops)
        result = new_node(result, tapes)                        # Q: but here tapes is a Set, not a dict or pair array?
        for (tape, argnum, parent) in ops
            gradfun = gradmaker(p, argnum, result, args, kwargs)
            rnode = result.tapes[tape]                          # rnode is a tape?
            push!(rnode.parent_grad_ops, (gradfun, parent))     # Q: who defined parent_grad_ops?
        end
    end
    return result
end


# TODO: what is partial, define it?
# TODO: what is primitive with aux, do we need it?
# TODO: NotImplemented, NoDerivativeNode?
# TODO: ReverseNode, with parent_grad_ops method?

# At this point we get the wrong contents in Node.tapes
# Time to implement Node and ReverseNode correctly.

function backward_pass(start_node, end_node, tape)
    if !isa(end_node, Node) || !haskey(end_node.tapes, tape)
        warn("Output seems independent of input. Returning zero gradient.")
        return zeros_like(start_node)
    end
    # if !isa(end_node, FloatNode)  ## we don't have float nodes
    #     try
    #         end_node = FloatNode.cast(end_node, 1.0)
    #     catch TypeError
    #         #TODO: throw TypeError(
    #         #                 "Output type $(typeof(end_node.value)) can't be cast to float. "
    #         #                 "Function grad requires a scalar-valued function. "
    #         #                 "Try jacobian or elementwise_grad.")
    #     end
    # end
    for node in tape
        node.outgrads = []
    end
    end_node.tapes[tape].outgrads = [1.0]

    complete!(tape)
    cur_outgrad = nothing
    for node in tape[end-1:-1:1]                                # note the end-1 because we pushed nothing to complete
        if !isempty(node.outgrads)
            cur_outgrad = sum_outgrads(node)
            # TODO: @assert (type(new_node(getval(cur_outgrad))) == node.node_type) "Types are {0} and {1}" # TODO:.format(type(new_node(getval(cur_outgrad))), node.node_type)
            for (gradfun, parent) in node.parent_grad_ops
                # Q: Should not be necessary with a single node type:
                # og = cast_to_node_type(gradfun(cur_outgrad), parent.node_type, parent.node_value)
                og = gradfun(cur_outgrad)
                push!(parent.outgrads, og)
            end
        end
    end
    return cur_outgrad
end

# Type signatures for gradients:
# f: X->Y
# Primitive(f)=P: Node(X)->Node(Y)?
# gradmaker: (Y,X)->(dY->dX)
# grad: P -> (X->dX)
# defgrad: gradmaker -> Void (puts gradmaker in grads)

# Primitive definition:
# Q: should we as for gradients right away?
# Q: can we do gradients without higher order functions?
# Q: we need a Node method, what else do we absolutely need?

psin = Primitive(sin)
pcos = Primitive(cos)
sin(x::Node)=psin(x)
cos(x::Node)=pcos(x)
# Instead of recording inputs and outputs explicitly, we hide it in closures for every call?
# We store pointers to all inputs and outputs!
defgrad(psin, (y,x)->(dy->dy * pcos(x)))                         # Q: why * and not /?
defgrad(pcos, (y,x)->(dy->-dy * psin(x)))

(start_node, end_node, tape) = forward_pass(sin, [1.0,], [], 1)
@show start_node # value:1.0, tapes:(tape=>tape[1])
@show end_node   # value:0.84, tapes:(tape=>tape[2])
display(tape)
# type ReverseNode; parent_grad_ops; outgrads; node_type; node_value; end
# tape[1]: ReverseNode(Any[],Any[],Node,1.0)                                                                       
# tape[2]: ReverseNode(Any[((anonymous function),ReverseNode(Any[],Any[],Node,1.0))],Any[],Node,0.8414709848078965)

gsin = grad(sin)
@show sin(1.0)
@show gsin(1.0)

:ok
