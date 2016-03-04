"""
The @knet macro defines a new knet function:

    @knet function wbf(x; f=relu, o...)
        y = wdot(x; o...)
        z = bias(y; o...)
        return f(z; o...)
    end

Once a knet function is defined it can be:

    (1) Used as an operator in other knet function definitions
    (2) Compiled into a knet model (Net) using the compile function
"""
macro knet(f)
    (fname, fargs, fpars, fbody) = _comp_parse_def(f)
    esc(Expr(:call,Kfun.kdef,Expr(:quote,fname),Expr(:quote,f)))
end

"""
compile(fname::Symbol; o...) compiles the knet function given by fname
into a knet model, i.e. a Net object.  fname should be a symbol
(i.e. use compile(:lstm), not compile(lstm)).  Optional keyword
arguments (o...) can be used to pass initialization parameters to the
compiler.  Knet function names in keyword args should also be escaped
with a colon.  For example:

    net = compile(:wbf; f=:sigm, out=100)
"""    
function compile(fname::Symbol; o...)
    @dbg println((:compile,:fname,fname,:o,o))
    isdefined(Kfun, fname) || error("$fname not defined as a knet function, use @knet in function declaration")
    prog = _comp(Kfun.(fname); o...)
    @dbg println((:compile,:prog,prog))
    Net(prog)
    # inst = _comp_inst(prog)
    # @dbg println((:compile,:inst,inst))
    # Net(inst)
end

function _comp(f::Expr; o...)
    @dbg println((:_comp1,:f,f,:o,o))
    (fname, fargs, fpars, fbody) = _comp_parse_def(f)
    expr = Expr(:block)
    for x in fargs; push!(expr.args, :($x=input())); end
    append!(expr.args, fbody.args)
    locals = [x=>x for x in _comp_locals(fbody)]
    @dbg println((:_comp1,:locals,locals))
    fpars = _comp_fpars(f, o)
    cond = Expr(:&&)  # A 0-arg && evaluates to true
    prog = _comp(expr, locals, fpars, cond)
    @dbg println((:_comp1,:return,prog))
    return prog
end

function _comp{T<:Op}(f::Type{T}; o...)
    @dbg println((:_comp2,:f,f,:o,o))
    feval = f(;o...)
    fargs = [ symbol("x$i") for i in 1:ninputs(feval) ]
    fcall = Expr(:call, feval, fargs...)
    expr = Expr(:block)
    for x in fargs; push!(expr.args, :($x=input())); end
    push!(expr.args, Expr(:return, fcall))
    locals = [x=>x for x in _comp_locals(expr)]
    @dbg println((:_comp2,:locals,locals))
    prog = _comp(expr, locals, Dict(), Expr(:&&))
    @dbg println((:_comp2,:return,prog))
    return prog
end

# _comp compiles expr in the context defined by name, value, and cond.
# It returns a Vector{Reg} array of instructions.
# name is a Dict{Symbol,Symbol} that provides name substitution rules.
# value is a Dict{Symbol,Any} that gives values for variables.
# cond is an Expr that gives the current condition.
#
# Example:
# 
# @knet function wf(x; f=relu, o...)
#     y = wdot(x; o...)
#     return f(y; o...)
# end
#
# compile(:wf; out=10, f=:sigm)
#
# calls _comp with:
# expr: body of wf prepended with :(x=input())
# name: {:x=>:x, :y=>:y, :return=>:return} preserving the local variable names of the top level call
#       name is used for renaming variables that are lhs in assignments, and positional arguments.
#       any local variable not found in name will be replaced with a gensym when compiling the body.
# value: {:f=>:sigm, :o=>[(:out,10)]} these values should be used while compiling the function body
#        value is for evaluating variables that are function names and keyword arguments.
# cond: :($(Expr(:&&))) the initial cond is an empty conjunction which evaluates to true

function _comp(expr,name::Dict,value::Dict,cond::Expr)
    @dbg println((:_comp,:expr,expr,:name,name,:value,value,:cond,cond))
    p = (isa(expr,LineNumberNode) ? Any[] :
         !isa(expr,Expr) ? error("expecting expression got $expr") :
         expr.head == :block ? mapreduce(x->_comp(x,name,value,cond), append!, expr.args) :
         expr.head == :if ? _comp_if(expr, name, value, cond) :
         expr.head == :return ? _comp(Expr(:(=), :return, expr.args[1]),name,value,cond) : # return y -> return = y
         expr.head != :(=) ? error("expecting assignment expression got $expr") :
         !isa(expr.args[1], Symbol) ? error("lhs should be a symbol in $expr") :
         isa(expr.args[2], Symbol) ? _comp(Expr(:(=), expr.args[1], Expr(:call,:copy,expr.args[2])),name,value,cond) : # y=x -> y=copy(x)
         !isa(expr.args[2], Expr) || expr.args[2].head != :call ? error("rhs should be a function call in $expr") :
         _comp_assignment(expr, name, value, cond) # y=f(x...;o...)
         )
    @dbg println((:_comp, :return, p))
    return p
end

# Continuing the last example, since the body is a block, _comp calls
# itself on each line recursively appending the results.  At some
# point we get to a typical line like :(y=wdot(x;o...))  which calls
# _comp_assignment with the name, value, cond given above.

# _comp_assignment :(y=wdot(x;o...)) has to create a new name2, value2, cond2 to compile the body of wdot.
# (1) eval the name of the function (wdot) in the value dict for possible replacement (it could be a variable f), then the Kfun global environment (all knet functions must be defined there).
# (2) lookup the name of lhs (y) in the name dict, creating a gensym for it if not found.  Set name2[:return] to this name.
# (3) compile any arg expressions, adding their instructions to the program, and generating gensyms for their return variables. Add these as {par=>arg} mappings to name2.
# (4) lookup the names of the args (x) in the name dict, creating gensyms for them if not found, adding {par=>arg} mappings to name2.
# (5) evaluate keyword argument values in the call using (a) value dict, (b) Kfun table, (c) julia environment.
# (6) construct the value dict considering the function definition of wdot and the output of #5.

function _comp_assignment(expr::Expr,name::Dict,value::Dict,cond::Expr)
    @dbg println((:_comp_assignment,:expr,expr,:name,name,:value,value,:cond,cond))
    prog = Reg[]
    (f,fargs,fpars) = _comp_parse_call(expr.args[2])

    haskey(value, f) && (f=value[f])
    # haskey(_KENV, f) || error("$f is not a knet function.")
    # feval = _KENV[f]
    # feval = _comp_eval(f, value)
    feval = eval(Kfun, f)
    @dbg println((:_comp_assignment,:feval,feval))

    y = expr.args[1]
    yname = get!(name, y, gensym(y))
    @dbg println((:_comp_assignment,:yname,yname))

    xname = map(fargs) do x
        if isa(x,Symbol)
            get!(name,x,gensym(x))
        elseif isa(x,Expr)
            # TODO: use the same reg if possible for register sharing
            tmp = gensym(:tmp)
            name[tmp] = tmp
            append!(prog, _comp(Expr(:(=), tmp, x), name, value, cond))
            tmp
        else
            error("Malformed positional argument $x in $expr")
        end
    end
    xname = convert(Vector{Symbol}, xname)
    @dbg println((:_comp_assignment,:xname,xname))

    odict = Dict{Symbol,Any}()
    for k in fpars              # this is the fpars from the call, not the definition
        if k.head == :kw
            odict[k.args[1]] = _comp_eval(k.args[2], value)
        elseif k.head == :(...)
            for (a,b) in _comp_eval(k.args[1], value)
                odict[a] = b
            end
        else
            error("Malformed keyword argument $k in $expr")
        end
    end
    @dbg println((:_comp_assignment,:odict,odict))

    if isa(feval, Op)
        # This happens when compiling a primitive Op directly, kwargs already taken into account in comp2
        push!(prog, Reg(feval, yname, xname, cond))
    elseif isa(feval, DataType) && (feval <: Op)
        op = feval(; odict...)
        push!(prog, Reg(op, yname, xname, cond))
    else
        isa(feval, Function) && (feval = feval(; odict...)) # This allows macros like repeat
        isa(feval, Expr) || error("expecting Op or Expr got $f")
        name2 = _comp_fargs(feval, xname, yname)
        value2 = _comp_fpars(feval, odict)
        fbody = feval.args[2]
        append!(prog, _comp(fbody, name2, value2, cond))
    end
    @dbg println((:_comp_assignment,:return,prog))
    return prog
end

function _comp_if(expr::Expr,name::Dict,value::Dict,cond::Expr)
    @dbg println((:_comp_if,:expr,expr,:name,name,:value,value,:cond,cond))
    p = append!(_comp(expr.args[2],name,value,Expr(cond.head,cond.args...,expr.args[1])),
                length(expr.args) == 2 ? Any[] :
                _comp(expr.args[3],name,value,Expr(cond.head,cond.args...,Expr(:call,:!,expr.args[1]))))
    @dbg println((:_comp_if,:return,p))
    return p
end

function _comp_locals(ex)       # TODO: should we ignore conditionals, which are global?
    @dbg println((:_comp_locals,:ex,ex))
    l = (isa(ex, Symbol) ? Any[ex] :
         isa(ex, Number) ? Any[] :
         isa(ex,LineNumberNode) ? Any[] :
         !isa(ex, Expr) ? error("Expected Expr got $ex") :
         ex.head == :parameters ? Any[] :
         ex.head == :kw ? Any[] :
         ex.head == :return ? mapreduce(_comp_locals, append!, Any[:return], ex.args) :
         ex.head == :call ? mapreduce(_comp_locals, append!, Any[], ex.args[2:end]) :
         mapreduce(_comp_locals, append!, Any[], ex.args))
    @dbg println((:_comp_locals,:return,l))
    return l
end

# Given an expression or symbol and a dictionary of symbols, evaluate the 
# expression in an environment defined by the dict, and Julia.
function _comp_eval(s,d::Dict)
    @dbg println((:_comp_eval,:s,s,:d,d))
    a = Expr(:let,Expr(:block,s))
    for (k,v) in d
        isa(v,Symbol) && (v=Expr(:quote,v)) # this is to prevent over-evaluation of (f=:sigm)
        push!(a.args, Expr(:(=),k,v))
    end
    ev = eval(current_module(), a)
    @dbg println((:_comp_eval,:return,ev))
    return ev
end


# Given a function definition, an array of argument symbols, x, and a
# return symbol, y, return a name dictionary that should be used when
# compiling the body of the function.
function _comp_fargs(f::Expr, x::Array, y::Symbol)
    @dbg println((:_comp_fargs,:f,f,:x,x,:y,y))
    (fname, fargs, fpars) = _comp_parse_call(f.args[1])
    length(fargs)==length(x) || error("parameter mismatch [$fargs] [$x]")
    fdict = Dict{Symbol,Symbol}()
    for i=1:length(x)
        (isa(x[i],Symbol) && isa(fargs[i],Symbol)) || error("parameter not symbol [$fargs] [$x]")
        fdict[fargs[i]] = x[i]
    end
    fdict[:return] = y
    @dbg println((:_comp_fargs,:return,fdict))
    return fdict
end

# Given a function definition and an array of keyword arguments found
# in the function call, returns a dictionary of keyword arguments that
# should be passed to the body of the function.  I tried writing my
# own function for this, but it is difficult to be consistent with the
# Julia implementation, e.g. having a kwarg value be in scope for a
# later kwarg etc.  It is easier to just construct a Julia function
# and see what it does.  Unfortunately anonymous functions do not
# currently accept kwargs
# (https://github.com/JuliaLang/julia/issues/2773), so we construct a
# named function.

function _comp_fpars(f::Expr, o)
    @dbg println((:_comp_fpars,:f,f,:o,o))
    (fname, fargs, fpars) = _comp_parse_call(f.args[1]) # these are from the original function definition
    ftemp = :_comp_fpars_tmp                            # constructing a new function definition
    fhead = Expr(:call, ftemp)
    !isempty(fpars) && push!(fhead.args, Expr(:parameters, fpars...)) # including fpars, excluding fargs
    fvars = map(s->Expr(:(=>), QuoteNode(s.args[1]), s.args[1]), fpars)
    fbody = Expr(:block, Expr(:call, :Dict, fvars...))
    fdefn = Expr(:function, fhead, fbody) # the new function returns a dictionary of fpars and their values
    # If we use Kfun instead of current_module to capture knet
    # functions, things go awry, e.g. a kwarg (k=a*b) ends up calling
    # Dot instead of regular multiply.  It is best not to mix Kfun and
    # Julia.
    eval(current_module(),fdefn)
    fdict = eval(current_module(),ftemp)(;o...)
    @dbg println((:_comp_fpars,:return,fdict))
    return fdict
end

function _comp_parse_def(f)
    @dbg println((:_comp_parse_def,:f,f))
    (isa(f,Expr) &&
     (f.head == :function || f.head == :(=)) &&
     (length(f.args) == 2) &&
     (f.args[1].head == :call) &&
     (f.args[2].head == :block)) ||
    error("Expected function definition got $f")
    (fhead, fbody) = f.args
    (fname, fargs, fpars) = _comp_parse_call(fhead)
    @dbg println((:_comp_parse_def,:return,:fname,fname,:fargs,fargs,:fpars,fpars,:fbody,fbody))
    (fname, fargs, fpars, fbody)
end

function _comp_parse_call(s)
    @dbg println((:_comp_parse_call,:s,s))
    (isa(s,Expr) && s.head == :call) ||
    error("Expected function call got $s")
    f = s.args[1]
    a = s.args[2:end]           # Use [:] notation to create a copy
    f == :+  ? _comp_parse_add(a...) :
    f == :.+ ? _comp_parse_badd(a...) :
    f == :-  ? _comp_parse_sub(a...) :
    f == :.- ? _comp_parse_bsub(a...) :
    f == :*  ? _comp_parse_mul(a...) :
    f == :.* ? _comp_parse_bmul(a...) :
    f == :/  ? _comp_parse_div(a...) :
    f == :./ ? _comp_parse_bdiv(a...) :
    _comp_parse_fcall(f,a)
end

# what if we have expression instead of symbol: define a compound type SymEx
# what if we have symbol instead of number (from compiler kwargs): can't happen, value subst does not take place with variables, only op args and func names

typealias SymEx Union{Symbol,Expr}

_comp_parse_add(x)=error("+$x")
_comp_parse_add(x,y)=_comp_parse_badd(x,y)
_comp_parse_add(x...)=_comp_parse_call(_comp_binarize(:+,x...))
_comp_parse_badd(x::SymEx,y::SymEx)=(:.+, Any[x,y], Any[])
_comp_parse_badd(x::Number,y::SymEx)=(:axpb, Any[y], Any[Expr(:kw,:b,x)])
_comp_parse_badd(x::SymEx,y::Number)=(:axpb, Any[x], Any[Expr(:kw,:b,y)])
_comp_parse_badd(x...)=error(".+$x")

_comp_parse_sub(x...)=_comp_parse_bsub(x...)
_comp_parse_bsub(x::SymEx,y::SymEx)=(:.+, Any[x,:(axpb($y;a=-1))], Any[])
_comp_parse_bsub(x::Number,y::SymEx)=(:axpb, Any[y], Any[Expr(:kw,:a,-1), Expr(:kw,:b,x)])
_comp_parse_bsub(x::SymEx,y::Number)=(:axpb, Any[x], Any[Expr(:kw,:b,-y)])
_comp_parse_bsub(x::SymEx)=(:axpb, Any[x], Any[Expr(:kw,:a,-1)])
_comp_parse_bsub(x...)=error(".-$x")

_comp_parse_mul(x::SymEx,y::SymEx)=(:*, Any[x,y], Any[])
_comp_parse_mul(x,y)=_comp_parse_bmul(x,y)
_comp_parse_mul(x...)=_comp_parse_call(_comp_binarize(:*,x...))
_comp_parse_mul(x)=error("*$x")
_comp_parse_bmul(x::SymEx,y::SymEx)=(:.*, Any[x,y], Any[])
_comp_parse_bmul(x::Number,y::SymEx)=(:axpb, Any[y], Any[Expr(:kw,:a,x)])
_comp_parse_bmul(x::SymEx,y::Number)=(:axpb, Any[x], Any[Expr(:kw,:a,y)])
_comp_parse_bmul(x...)=error(".*$x")

_comp_parse_div(x::SymEx,y::SymEx)=error("$x/$y")
_comp_parse_div(x,y)=_comp_parse_bdiv(x,y)
_comp_parse_div(x...)=error("/$x")
_comp_parse_bdiv(x::SymEx,y::SymEx)=(:.*, Any[x,:(axpb($y;p=-1))], Any[])
_comp_parse_bdiv(x::Number,y::SymEx)=(:axpb, Any[y], Any[Expr(:kw,:a,x),Expr(:kw,:p,-1)])
_comp_parse_bdiv(x::SymEx,y::Number)=(:axpb, Any[x], Any[Expr(:kw,:a,1/y)])
_comp_parse_bdiv(x...)=error("./$x")

_comp_binarize(f,x,y)=Expr(:call,f,x,y)
_comp_binarize(f,x,y,z,t...)=_comp_binarize(f,Expr(:call,f,x,y),z,t...)

function _comp_parse_fcall(f, a)
    x = Any[]
    o = Any[]
    for ai in a
        !isa(ai,Expr) ? push!(x,ai) :
        ai.head==:parameters ? append!(o, ai.args) :
        ai.head==:kw ? push!(o, ai) :
        push!(x, ai)
    end
    @dbg println((:_comp_parse_fcall,:return,:fname,f,:fargs,x,:fpars,o))
    (f, x, o)
end

function test_compiler()
    for n in names(Kfun,true)
        isa(Kfun.(n),DataType) || isa(Kfun.(n),Expr) || continue
        println(n)
        compile(n; out=10, f=:tanh)
    end
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


# The core of the compiler is the compilation of a function call.  If
# the function is a primitive, this gets compiled into a single
# instruction.  If it is a compound, then the body of the function
# with certain transformations gets compiled into an array of
# instructions.

# The compilation of the function call takes place in a lexical
# environment.  There are several components of this environment:

# 1. The target variable.  The function call can be the rhs of an
# assignment statement, an internal call in a compound statement, or
# we may be compiling the top level function as a model.  In each case
# a target symbol should be provided to _comp_call.  Any return
# statement in the body of the function will be turned into an
# assignment to this target symbol.  But see #3.

# 2. The function name.  This could be a symbol in _kenv, which is a
# table of knet functions and their names, or a kwarg symbol from the
# parent function, in which case it should be evaluated.  The kwarg
# takes precedence.

# 3. The positional arguments.  These are either symbols or other
# calls.  To compile other calls we gensym target variables and insert
# their instructions preceding the current function call.  Once each
# argument turns into a symbol, these symbols will need to replace the
# parameters of the function call in the body.  Recursively this means
# we need a symbol map passed to _comp_call.  The target variable can
# also be in this map as the value of :return.

# 4. The keyword arguments.  These come as symbol=expr pairs,
# optionally followed by a o... splat.  The expr can be a kenv symbol,
# or a julia expression which should be evaluated in that order.
# These symbols should go into a symbol table and be used while
# compiling the body of the function.

# What is the simplest way to do this?  How many separate symtables do
# we need?  Can we combine symbol lookup and expr eval?

# eval args until all are symbols => we need a real _comp_call?
# should we do syntactic xforms?


# fargs or fpars could be empty depending on whether the user uses semicolon
# they may also have plain symbols or splats, which we should warn the user about
# fname could be a primitive op or a kfun, but it should be stored in _kenv in either case
# f = get(_kenv, fname, nothing)
# f == nothing && error("$fname not defined as a knet function")
# # in either case we should be able to compile it
# @show global tmp = 
# fcomp = eval(current_module(), tmp)

# user defines kfun using @knet function foo...
# user compiles kfun calling @knet kfun(a=1,b=2...)
# this should return a net.
# user uses kfun in other code using kfun(x,y;a=1,b=2...)
# the two usages may be confusing.
# we could just provide compile(:kfun; args...) instead of the knet macro

# function _knet(f)
#     (fname, fargs, fpars) = _comp_parse_call(f)
#     haskey(_kenv, fname) || error("$fname not defined as a knet function")
#     isempty(fpars) && ((fargs,fpars)=(fpars,fargs))
#     isempty(fargs) || error("Usage: @knet f(a=1,b=2,...)")
#     ex = Expr(:call,:comp,Expr(:parameters,fpars...),Expr(:ref,:_kenv,QuoteNode(fname)))
#     fcomp = eval(current_module(), ex)
# end

# macro kdef(f)
#     _kdef(f)
# end

# # """
# # The @knet macro compiles a knet function as a model:

# #     net = @knet foo(a=1,b=2)

# # compiles the knet function foo as a model with initialization
# # parameters `a=1` and `b=2` and assigns the result to the Julia
# # variable `net`.
# # """
# macro knet(f)
#     if f.head==:function
#         _kdef(f)
#     elseif f.head == :(=)
#         _kdef(f)
#     elseif f.head == :call
#         _knet(f)
#     end
# end

    # inputs = filter(x->isa(x.op,Input), op)
    # params = filter(x->isa(x.op,Par), op)

# """
# _kops lists primitive operations and their knet names.  The primitive
# operations (capitalized in the code) are represented by types that are
# subtypes of Op.
# """
# _kops = [(:add,Add),(:+,Add),   # TODO: catch add with alpha/beta
#          (:arr,Arr),            
#          (:axpb,Axpb),          # TODO: catch axpb
#          (:conv,Conv),
#          (:dot,Dot),(:*,Dot),
#          (:input,Input),
#          (:logp,Logp),
#          (:mul,Mul),(:.*,Mul),  # TODO: define div,sub
#          (:nce,Nce),
#          (:par,Par),
#          (:pool,Pool),
#          (:relu,Relu),
#          (:rnd,Rnd),
#          (:sigm,Sigm),
#          (:soft,Soft),
#          (:tanh,Tanh)]

#     feval = _comp_func(f, value)
# # TODO: not sure if we need this special treatment:
# # TODO: this forces quotation f=:relu, is there another way?
# function _comp_func(s,d::Dict)
#     @dbg println((:_comp_func,:s,s,:d,d))
#     f = s
#     haskey(d, f) && (f = d[f])
#     haskey(_KENV, f) && (f = _KENV[f])
#     @dbg println((:_comp_func,:return,f))
#     return f
# end

# macro knet(f)
#     (fname, fargs, fpars, fbody) = _comp_parse_def(f)
#     esc(Expr(:(=),Expr(:ref,:_KENV,Expr(:quote,fname)),Expr(:quote,f)))
# end

# function _comp_fpars(f::Expr, o)
#     @dbg println((:_comp_fpars,:f,f,:o,o))
#     (fname, fargs, fpars) = _comp_parse_call(f.args[1]) # these are from the function definition
#     slurp = (!isempty(fpars) && isa(fpars[end],Expr) && fpars[end].head == :(...) ? pop!(fpars).args[1] : nothing)
#     fdict = Dict{Symbol,Any}()
#     for k in fpars
#         (isa(k, Expr) && isa(k.args[1], Symbol) && k.head == :kw) || error("Malformed keyword argument $k")
#         fdict[k.args[1]] = eval(Kfun, k.args[2])
#         # (isa(k.args[2],Symbol) && haskey(_KENV,k.args[2]) ? _KENV[k.args[2]] :
#         #  eval(current_module(), k.args[2]))
#     end
#     if slurp != nothing
#         fdict[slurp] = Any[]
#     end
#     for (k,v) in o
#         if haskey(fdict, k)
#             fdict[k] = v
#         elseif slurp != nothing
#             push!(fdict[slurp], (k,v))
#         else
#             error("Unrecognized keyword argument \"$k\"")
#         end
#     end
#     @dbg println((:_comp_fpars,:return,fdict))
#     return fdict
# end

# the compiler turns expressions into low level instruction sequences.
# + variable renaming in function calls.
# + inheriting conditions
# + inheriting keyword arguments
# + replacing returns with variables
# + recording of arg names at top level
# + preserve local and arg variable names of top level function
# + representation of primitive ops
# + ninputs problem: look up the definition or ask the Op
# + can we compile primitive operators into net: why not
# + separate env table for knet ops and funs: cleaner but cannot pass name to Net or comp, or f=relu type kwarg unless we write macros.
# + adapt to the new _kenv structure: need next pointers?
# + compound operations
# + arithmetic operators
# + multiple functions sharing registers vs one with conditionals
# + instead of lots of arrays in net, have an Instruction type with the necessary fields: fix the rest of the code in net/
# + wf and f=:sigm not working. the dict that comes to _comp_eval should splat :o.
# + it works in recursive call from lstm to add2


:ok
