"@knet takes a function definition and assigns its AST to the function name."
macro knet(f); esc(Expr(:(=),f.args[1].args[1],Expr(:quote,f))); end

# the compiler turns expressions into low level instruction sequences.
# - variable renaming in function calls.
# - inheriting conditions
# - inheriting keyword arguments
# - replacing returns with variables

function comp(f::Expr; o...)
    ((f.head == :function || f.head == :(=)) &&
     (length(f.args) == 2) &&
     (f.args[1].head == :call) &&
     (f.args[2].head == :block)) ||
    error("the compiler expects a function definition.")
    returnvar = gensym(:return)
    _comp(f.args[2]; returnvar=returnvar)
end

function _comp_block(s::Expr; o...)
    mapreduce(x->_comp(x;o...), append!, s.args)
end

function _comp_if(s::Expr; cond=true, o...)
    append!(_comp(s.args[2]; cond=(cond==true ? s.args[1] : Expr(:&&, cond, s.args[1])), o...),
            length(s.args) == 2 ? Any[] :
            _comp(s.args[3]; cond=(cond==true ? Expr(:!,s.args[1]) : Expr(:&&, cond, Expr(:!,s.args[1]))), o...))
end

function _comp_return(s::Expr; returnvar=nothing, o...)
    @assert returnvar != nothing
    _comp(Expr(:(=), returnvar, s.args[1]); o...)
end

function _comp_assign(s::Expr; o...)
    @show (:assign, s, o)
    Any[]
end

# pass down pars with appropriate overrides and evaluation
# variable replacement in subroutine calls
# recording of arg names at top level
# representation of primitive ops
# ninputs problem

function _comp(s::Expr; o...)
    if s.head == :block
        _comp_block(s; o...)
    elseif s.head == :if
        _comp_if(s; o...)
    elseif s.head == :return
        _comp_return(s; o...)
    elseif s.head == :(=)
        _comp_assign(s; o...)
    else
        @show (:unknown, s)
    end
end

_comp(s::LineNumberNode; o...)=Any[]
_comp(s; o...)=(@show typeof(s); Any[])

# comp(f; o...) => o: compiler options
# forw(f,x; o...) => o: runtime options
# so we can do forw(f,x; training=true, encoding=false) etc.
# we already do this!
# so forw has to evaluate conditions with respect to o... and we are all set!
# do we compile with the @knet macro or the comp function?

# The compiler:
_knet(x::Expr)=(x.head == :block ? _knet_bloc(x.args) : x.head == :(=) ? _knet_assn(x.args...) : error())
_knet_bloc(x::Array)=mapreduce(_knet, append!, x)
_knet_assn(s::Symbol, x::Expr)=(x.head == :call ? _knet_call(x.args...,s) : error())
_knet_call(f, p::Expr, o...)=(p.head == :parameters ? _knet(eval(current_module(), Expr(:call, f, p, map(QuoteNode, o)...))) : error())
_knet_call(f, o...)=_knet(eval(current_module(),Expr(:call,f,map(QuoteNode, o)...)))
_knet(x::Tuple)=(isa(x[1],Op) ? Any[x] : error())
_knet(::LineNumberNode)=Any[]



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
