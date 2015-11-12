"@knet takes a function definition and assigns its AST to the function name."
macro knet(f); esc(Expr(:(=),f.args[1].args[1],Expr(:quote,f))); end

# the compiler turns expressions into low level instruction sequences.
# - variable renaming in function calls.
# - inheriting conditions
# - inheriting keyword arguments
# - replacing returns with variables
# - variable replacement in subroutine calls
# - recording of arg names at top level
# - representation of primitive ops
# - ninputs problem
# - separate env table for knet ops and funs

function comp(f::Expr; o...)
    ((f.head == :function || f.head == :(=)) &&
     (length(f.args) == 2) &&
     (f.args[1].head == :call) &&
     (f.args[2].head == :block)) ||
    error("the compiler expects a function definition.")
    (fhead, fbody) = f.args
    (fname, fargs, fpars) = _comp_call(fhead)
    kwargs = _comp_kwargs(f, o)
    args = [x=>x for x in fargs]
    args[:return] = :return
    _comp(fbody, args, kwargs, Expr(:&&))
    # returnvar = gensym(:return)
    # cond = Expr(:&&)
    # kwargs = _comp_kwargs(f, o)
    # instructions = _comp(fbody, returnvar, cond, kwargs)
end

function _comp(expr,name::Dict,value::Dict,cond::Expr)
    if isa(expr,LineNumberNode)
        Any[]
    elseif !isa(expr,Expr)
        error("expecting expression got $expr")
    elseif expr.head == :block
        mapreduce(x->_comp(x,name,value,cond), append!, expr.args)
    elseif expr.head == :return
        _comp(Expr(:(=), :return, expr.args[1]),name,value,cond)
    elseif expr.head == :if
        append!(_comp(expr.args[2],name,value,Expr(cond.head,cond.args...,expr.args[1])), length(s.args) == 2 ? Any[] :
                _comp(expr.args[3],name,value,Expr(cond.head,cond.args...,Expr(:!,expr.args[1]))))
    elseif expr.head != :(=)
        error("expecting assignment expression got $expr")
    elseif !isa(expr.args[1], Symbol)
        error("lhs should be a symbol in $expr")
    elseif isa(expr.args[2], Symbol) # y=x
        _comp(Expr(:(=), expr.args[1], Expr(:call,:_copy,expr.args[2])),name,value,cond) # x=y -> x=copy(y)
    elseif !isa(expr.args[2], Expr) || expr.args[2].head != :call
        error("rhs should be a function call in $expr")
    else                        # y=f(x...;o...)
        y = expr.args[1]
        (f,x,o) = _comp_call(expr.args[2])

        yname = get!(name, y, gensym(y))
        feval = _comp_eval(f, value)
        xname = map(x) do v
            # TODO: evaluate x's instead of insisting them to be symbols
            isa(v,Symbol) ? get!(name,v,gensym(v)) :
            error("positional argument $v should be a symbol in $expr")
        end
        odict = Dict{Symbol,Any}()
        for k in o
            if k.head == :kw
                odict[k.args[1]] = _comp_eval(k.args[2], value)
            elseif k.head == :(...)
                for (a,b) in _comp_eval(k.args[1], value)
                    odict[a] = b
                end
            else
                error("Malformed keyword argument $k")
            end
        end
        if isa(feval, DataType) && (feval <: Op)
            op = feval(; odict...)
            Any[(cond, yname, op, xname...)]
        elseif isa(feval, Expr)
            kwargs = _comp_kwargs(feval, odict)
            args = _comp_args(feval, xname, yname)
            fbody = feval.args[2]
            _comp(fbody, args, kwargs, cond)
        else
            error("expecting Op or Expr got $f")
        end
    end
end

"""
Given an expression or symbol and a dictionary of symbols, evaluate the 
expression in an environment defined by the dict.
"""
function _comp_eval(s,d::Dict)
    a = Expr(:let,Expr(:block,s))
    for (k,v) in d
        push!(a.args, Expr(:(=),k,v))
    end
    eval(current_module(), a)
end

function _comp_args(f::Expr, x::Array, y::Symbol)
    (fname, fargs, fpars) = _comp_call(f.args[1])
    length(fargs)==length(x) || error("parameter mismatch [$fargs] [$x]")
    fdict = Dict{Symbol,Symbol}()
    for i=1:length(x)
        (isa(x[i],Symbol) && isa(fargs[i],Symbol)) || error("parameter not symbol [$fargs] [$x]")
        fdict[fargs[i]] = x[i]
    end
    fdict[:return] = y
    return fdict
end

"""
Given a function definition and an array of keyword arguments, returns
a dictionary of keyword arguments that should be passed to the body of
the function.
"""        
function _comp_kwargs(f::Expr, o)
    (fname, fargs, fpars) = _comp_call(f.args[1])
    isempty(fpars) && return Any[]
    slurp = (isa(fpars[end],Expr) && fpars[end].head == :(...) ? pop!(fpars).args[1] : nothing)
    fdict = Dict{Symbol,Any}()
    for k in fpars
        (isa(k, Expr) && isa(k.args[1], Symbol) && k.head == :kw) || error("Malformed keyword argument $k")
        fdict[k.args[1]] = eval(current_module(), k.args[2])
    end
    if slurp != nothing
        fdict[slurp] = Any[]
    end
    for (k,v) in o
        if haskey(fdict, k)
            fdict[k] = v
        elseif slurp != nothing
            push!(fdict[slurp], (k,v))
        else
            error("Unrecognized keyword argument \"$k\"")
        end
    end
    return fdict
end

function _comp_call(s::Expr)
    f = s.args[1]
    x = s.args[2:end]
    o = !isempty(x) && isa(x[1],Expr) && x[1].head==:parameters ? shift!(x).args[1:end] : Any[]
    (f, x, o)
end

### DEAD CODE:

# # The compiler:
# _knet(x::Expr)=(x.head == :block ? _knet_bloc(x.args) : x.head == :(=) ? _knet_assn(x.args...) : error())
# _knet_bloc(x::Array)=mapreduce(_knet, append!, x)
# _knet_assn(s::Symbol, x::Expr)=(x.head == :call ? _knet_call(x.args...,s) : error())
# _knet_call(f, p::Expr, o...)=(p.head == :parameters ? _knet(eval(current_module(), Expr(:call, f, p, map(QuoteNode, o)...))) : error())
# _knet_call(f, o...)=_knet(eval(current_module(),Expr(:call,f,map(QuoteNode, o)...)))
# _knet(x::Tuple)=(isa(x[1],Op) ? Any[x] : error())
# _knet(::LineNumberNode)=Any[]



# ### Structure of an example function expression:
# #
# # function drop(x)
# #     if training
# #         r = rnd(; rgen=Bernoulli(0.5))
# #         return mul(r,x)
# #     elseif testing
# #         x = r
# #         return x
# #     else
# #         return x
# #     end
# # end
# #
# # (function[2], call[2], block[2])		# call for function header, block for body
# #   (call[2], :drop, :x)			# this is more complicated with keyword args
# #   (block[2], line, if[3])			# need to ignore line number elements
# #     (if[3], :training, block[4], block[2])	# if will have two args (cond,consequent) if no else part
# #       (block[4], line, =[2], line, return[1])
# #         (=[2], :r, call[2])
# #           (call[2],:rnd,parameters[1])
# #             (parameters[1],kw[2])		# keyword arguments following semicolon, see without semicolon
# #               (kw[2],:rgen,call[2])		# kw takes place of =
# #                 (call[2],:Bernoulli,0.5)
# #         (return[1], call[3])
# #           (call[3],:mul,:r,:x)
# #       (block[2], line, if[3])			# elseif turns into a simple if inside the else component
# #         (if, :testing, block[4], block[2])
# #           (block[4],line,=[2],line,return[1])
# #             (=[2],:x,:r)
# #             (return[1],:x)
# #           (block[2],line,return[1])
# #             (return[1],:x)

# # Julia parser output for function parameters:
# #
# # foo(x)	:call(foo,x)
# # foo(x,y)	:call(foo,x,y)
# # foo(x,y=0)	:call(foo,x,:kw(y,0))
# # foo(y=0)	:call(foo,:kw(y,0))
# # foo(;y=0)	:call(foo,:parameters(:kw(y,0)))
# # foo(x;y=0)	:call(foo,:parameters(:kw(y,0)),x)
# # foo(x,y=0;z=0):call(foo,:parameters(:kw(z,0)),x,:kw(y,0))
# # foo(x,y...)	:call(foo,x,:(...)(y))
# # foo(x;o...)	:call(foo,:parameters(:(...)(o)),x)
# # foo(x,y=0,z...;a,b=0,c...)	:call(foo,:parameters(a,:kw(b,0),:(...)(c)),x,:kw(y,0),:(...)(z))

# macro knet_old(f)
#     @assert f.head == :function "@knet macro requires a function definition as input."
#     @assert length(f.args) == 2
#     (fhead,fbody) = f.args

#     @assert fhead.head == :call
#     @assert length(fhead.args) >= 1
#     fname = fhead.args[1]
#     fargs = fhead.args[2:end]
#     if !isempty(fargs) && isa(fargs[1],Expr) && fargs[1].head == :parameters
#         fparameters = shift!(fargs)
#         fpars = map(fparameters.args) do k
#             @assert (isa(k,Expr) && isa(k.args[1], Symbol) && (k.head == :kw || (k.head == :(...) && k === fparameters.args[end]))) "Malformed keyword argument $k"
#             k.args[1]
#         end
#     else
#         fparameters = nothing
#         fpars = Any[]
#     end
#     @assert all(x->isa(x,Symbol), fargs) "Positional arguments must be symbols only, knet does not support default values or ... slurping before the semicolon."
#     @assert unique(fargs)==fargs "Positional arguments must be unique."

#     @assert fbody.head == :block
#     @show fvars = setdiff(_knet_vars(fbody), fargs)

#     freturn = gensym()
#     push!(fargs, freturn)
#     map!(s->Expr(:(::), s, :Symbol), fargs)
#     @show fargs

#     _knet_return(fbody, freturn)
    
    
#     # @show name = f.args[1].args[1]
#     # @show args = f.args[1].args[2:end]
#     # @show pars = _knet_pars(f)
#     # @show vars = _knet_vars(f)
#     # @assert !isempty(vars)

#     # @show head = Expr(:call, name, map(_knet_sym, [args; vars[end]])...)
#     # @show m = Expr(:macrocall, symbol("@gensym"), vars[1:end-1]...)
#     # @show q = Expr(:quote, _knet_esc(f.args[2], vcat(pars, vars)))
#     # body = Expr(:block, m, q)
#     # newf = Expr(:function, head, body)
#     # dump(newf,100)
#     # esc(newf)
# end

# "replace returns with assignment statements"
# function _knet_return(f::Expr, freturn::Symbol)
#     for a in f.args
#         if !isa(a, Expr)
#             continue
#         elseif a.head == :return
#             a.head = :(=)
#             unshift!(a.args, freturn)
#         else
#             _knet_return(a, freturn)
#         end
#     end
# end

# "find local variables that are assignment targets"
# function _knet_vars(f::Expr)
#     vars = Any[]
#     for a in f.args
#         if !isa(a, Expr)
#             continue
#         elseif a.head == :(=)
#             push!(vars, a.args[1])
#         else
#             append!(vars, _knet_vars(a))
#         end
#     end
#     return vars
# end

# function _knet_esc(x::Expr,v::Array)
#     if x.head == :line
#         x
#     elseif x.head == :kw
#         Expr(x.head, x.args[1], map(a->_knet_esc(a,v), x.args[2:end])...)
#     else
#         Expr(x.head, map(a->_knet_esc(a,v), x.args)...)
#     end
# end

# # TODO: add checks when s is not an element of v
# _knet_esc(s::Symbol,v::Array)=(in(s,v) ? Expr(:$, s) : s)
# _knet_esc(x,v::Array)=x

# _knet_sym(s::Symbol)=Expr(:(::), s, :Symbol)
# _knet_sym(x)=x

# # "positional and keyword arguments"
# # function _knet_pars(f::Expr)
# #     a = f.args[1].args[2:end]
# #     all(x->isa(x,Symbol), a) && return a
# #     @assert isa(a[1],Expr) && a[1].head == :parameters && all(x->isa(x,Symbol), a[2:end])
# #     vcat(a[2:end], map(kw->kw.args[1], a[1].args))
# # end

# # The compiler:
# _knet(x::Expr)=(x.head == :block ? _knet_bloc(x.args) : x.head == :(=) ? _knet_assn(x.args...) : error())
# _knet_bloc(x::Array)=mapreduce(_knet, append!, x)
# _knet_assn(s::Symbol, x::Expr)=(x.head == :call ? _knet_call(x.args...,s) : error())
# _knet_call(f, p::Expr, o...)=(p.head == :parameters ? _knet(eval(current_module(), Expr(:call, f, p, map(QuoteNode, o)...))) : error())
# _knet_call(f, o...)=_knet(eval(current_module(),Expr(:call,f,map(QuoteNode, o)...)))
# _knet(x::Tuple)=(isa(x[1],Op) ? Any[x] : error())
# _knet(::LineNumberNode)=Any[]



# """
# _knet(x) is a compiler that takes the expressions generated by a
# @knet function and outputs a list of machine instructions of the form:
# ```
# (Op, in1, in2, ..., out)
# ```
# where Op is a primitive operator, in1, in2, ..., out are symbols
# representing input and output registers.

# The whole compiler is 7 lines long!

# Example:

# (1) The user defines a @knet function:
# ```
# @knet function wb(x; o...)
#     y = wdot(x; o...)
#     z = bias(y; o...)
# end
# ```

# (2) The @knet macro turns this into a Julia function which takes
# input/output symbols and generates an expression sequence using these
# symbols.  The locals are replaced by gensyms:

# ```
# function wb(x::Symbol, z::Symbol; o...)
#     @gensym y
#     quote
#         \$y = wdot(\$x; \$o...)
#         \$z = bias(\$y; \$o...)
#     end
# end
# ```

# (3) The user calls a Net constructor, e.g. FNN(wb; out=100).  The net
# constructor (or recursively the compiler) runs the Julia function `wb`
# to get:

# ```
# julia> prog = wb(:a,:b; out=100)
# quote
#     ##y#8260 = wdot(a; Any[(:out,100)]...)
#     b = bias(##y#8260; Any[(:out,100)]...)
# end
# ```

# (4) This gets passed to the _knet compiler to get:
# ```
# julia> _knet(prog)
#  (Par((100,0),Gaussian(0,0.01),...),symbol("##w#8267"))
#  (Dot(),symbol("##w#8267"),:a,symbol("##y#8262"))
#  (Par((0,),Constant(0),...),symbol("##b#8270"))
#  (Add(),symbol("##b#8270"),symbol("##y#8262"),:b)
# ```

# (5) The Net constructor adds input expressions and replaces symbols with Ints to get:
# ```
# net.op = [ Input(), Par((100,0),...), Dot(), Par((0,),...), Add() ]
# net.inputs = [ [], [], [2,1], [], [4,3] ]
# ```
# """
# _knet


# """
# @knet macro -- This is what the user types:
# ```
# @knet function wdot(x; out=0, winit=Gaussian(0,.01), o...)
#     w = par(out,0; init=winit, o...)
#     y = dot(w, x)
# end
# ```
# This is what the @knet macro turns it into:
# ```
# function wdot(x::Symbol, y::Symbol; out=0, winit=Gaussian(0,.01), o...)
#     @gensym w
#     quote
#         \$w = par(\$out,0; init=\$winit, \$o...)
#         \$y = dot(\$w, \$x)
#     end
# end
# ```
# """
# :@knet


# :ok

    # fparameters contain the parameters in the function call
    # their values might be unevaluated
    # o... contains the parameters from the compiler call
    # the op constructor should be called with ?

# pass down pars with appropriate overrides and evaluation
# compiler options example:
# comp(f; hidden=100, lr=0.1)
# f = :(function f(x; hidden=0,o...); w = par(;out=hidden,o...); return dot(w,x); end)
# par should we called with out=100, lr=0.1
# if f(x; hidden=0) then par should be called with out=100 but no lr
# so the pars that get passed to the body of f is determined by the header of f
# if the header contains o..., then all pars are passed, overriding any defaults in f header

# function call example
# > w = par(; out=10, o...)
# function definition example
# > function wdot(x; out=10, o...)
# compiler call example
# > comp(wdot; out=10, o...)
# Differences:
# we can have o... in the middle in function call: par(; o..., out=10)
# we can have more than one splat: par(; o..., p..., out=10)
# Difference in terms of what should be passed as kwargs?
# Going from a comp call to function definition we excluded vars not in function kwarg list
# We treated the comp call as if it were a function call for wdot
# So it should be an error to have variables wdot does not expect

# 1. comp(wdot; hidden=10, lr=1) is called
# .. wdot(x; hidden=0, winit=Gaussian(0,.01), o...) is the signature of wdot
# 2. we construct a dictionary of keyword args for wdot from the comp call and wdot signature
# .. we pass this dictionary down to _comp...
# 3. in the body we have: w = par(; out=hidden, init=winit, o...)
# 4. we need to call par with (out=10,lr=1)
# this means the rhs of par keywords or par slurps need to be evaluated in an environment
# extended by our dict.

# comp(f; o...) => o: compiler options
# forw(f,x; o...) => o: runtime options
# so we can do forw(f,x; training=true, encoding=false) etc.
# we already do this!
# so forw has to evaluate conditions with respect to o... and we are all set!
# do we compile with the @knet macro or the comp function?

# function _comp_old(s::Expr, r::Symbol, c::Expr, d::Dict)
#     if s.head == :block
#         _comp_block(s, r, c, d)
#     elseif s.head == :if
#         _comp_if(s, r, c, d)
#     elseif s.head == :return
#         _comp_return(s, r, c, d)
#     elseif s.head == :(=)
#         _comp_assign(s, r, c, d)
#     else
#         @show (:unknown, s)
#     end
# end

# function _comp_block(s::Expr,r::Symbol,c::Expr,d::Dict)
#     mapreduce(x->_comp(x,r,c,d), append!, s.args)
# end

# function _comp_if(s::Expr,r::Symbol,c::Expr,d::Dict)
#     append!(_comp(s.args[2],r,Expr(c.head,c.args...,s.args[1]), d),
#             length(s.args) == 2 ? Any[] :
#             _comp(s.args[3],r,Expr(c.head,c.args...,Expr(:!,s.args[1])), d))
# end

# function _comp_return(s::Expr,r::Symbol,c::Expr,d::Dict)
#     _comp(Expr(:(=), r, s.args[1]),r,c,d)
# end

# function _comp_assign(s::Expr,r::Symbol,c::Expr,d::Dict)
#     @assert length(s.args)==2
#     (lhs,rhs) = s.args
#     @assert isa(lhs, Symbol)
#     if isa(rhs, Symbol)
#         @show (:assign_symbol, s,r,c,d); Any[]
#     else
#         @assert isa(rhs, Expr)
#         @assert rhs.head == :call
#         func = eval(current_module(), rhs.args[1])
#         if isa(func, Expr)
#             _comp_sub(s,r,c,d)
#         elseif isa(func, DataType) && func <: Op
#             _comp_op(s,r,c,d)
#         else
#             @show (:assign_unknown, s,r,c,d); Any[]
#         end
#     end
# end

# "compile: w = par(; out=10, o...)"
# function _comp_op(s::Expr,r::Symbol,c::Expr,d::Dict)
#     (lhs,rhs) = s.args
#     a = rhs.args
#     if length(a) > 1 && isa(a[2],Expr) && a[2].head == :parameters
#         fcall = a[1:2]
#         fargs = a[3:end]
#     else
#         fcall = a[1:1]
#         fargs = a[2:end]
#     end
#     @assert all(x->isa(x,Symbol), fargs) "Positional arguments must be symbols only, knet does not support default values or ... slurping before the semicolon."
#     @assert unique(fargs)==fargs "Positional arguments must be unique."
#     op = _comp_eval(Expr(:call,fcall...), d)
#     Any[tuple(c, lhs, op, fargs...)]
# end

# f (wdot) is a function definition.
# compile the body of f, but:
# returnvar needs to be lhs
# args in function def need to be replaced with fargs
# local vars in function def need to be gensym
# fpar dict needs to be handled
# we just need the instructions back.
# _comp the body
# we need two dictionaries: symbol replacement dict, and eval dict
# when we see y=wdot(x; o...) in the body we need to replace y, x, and o.
# when we see z = f(y; o...) find y and z in vars, f and o in dict
# so target and input vars in vars, all else eval
# initial: comp gets fname and kwargs
# kwargs matches kwpars in f definition, may have extras if o...

# problem symbols like conv and dot are defined in base and cannot be overwritten
# solution: make the compiler look them up in a special primitives table
# that special table can also include operators like '+' or '*'

# However if we do use a primitives table and do not define symbols 
# like dot in Julia then we can't pass them using comp(wbf;f=dot)

# """

# Compiling a single instruction::

#     y=f(x1,x2,...;o1=v1,o2=v2,on...)

# when f is a primitive operator or
# when the function definition for f is::

#     function f(z1,z2,...;p1=w1,p2=w2,pn...)    

# """
# _comp


