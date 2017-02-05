"""

gradcheck(f, w, x...; 
          gcheck=10,			# number of samples
          verbose=false,		# print stuff if true
          delta=atol=rtol=cbrt(eps(w)),	# tolerance params
          kwargs=[])			# keyword args for f

Numerically checks the gradient of `f(w,x...;o...)` with respect to
its first argument `w`, which could be a Number, Array, Tuple or Dict
and returns true if it passes.  `gcheck` largest entries from each
numeric array in gradient `dw=(grad(f))(w,x...;o...)` is compared to
its numerical estimate and a Warning is issued for comparisons that
fail.  If the output of f is not a number, gradcheck constructs a
scalar function by taking its dot product with a random vector.

"""
function gradcheck(f, w, x...; kwargs=[], o...)
    y = f(w, x...; kwargs...)
    if !isa(y,Number); f = gc_scalar(f); end
    g = grad(f)
    d = g(w, x...; kwargs...)
    if isa(w, Number)
        gc_number(d, f, w, x...; kwargs=kwargs, o...)
    elseif isa(w, KnetArray) || (isa(w, Array) && isbits(eltype(w)))
        gc_array(w, d, f, w, x...; kwargs=kwargs, o...)
    else
        k = gc_indices(w)
        pass = true
        for i in k
            pass &= gc_index(w, d, i, f, w, x...; kwargs=kwargs, o...)
        end
        return pass
    end
end

function gc_index(w, d, i, f, w0, x...; o...)
    if isa(w[i], Number)
        gc_array(w, d, f, w0, x...; icheck=i, o...)
    elseif isa(w[i],KnetArray) || (isa(w[i], Array) && isbits(eltype(w[i])))
        gc_array(w[i], d[i], f, w0, x...; o...)
    else
        k = gc_indices(w[i])
        pass = true
        for j in k
            pass &= gc_index(w[i], d[i], j, f, w0, x...; o...)
        end
        return pass
    end
end

function gc_array(w, d, f, worig, x...; gcheck=10, icheck=0, kwargs=[],
                  delta=dx(w), atol=dx(w), rtol=dx(w), verbose=false)
    if icheck > 0
        irange = (icheck:icheck)
    elseif length(w) <= gcheck
        irange = (1:length(w))
    elseif d == nothing
        irange = rand(1:length(w), gcheck)
    else
        irange = sortperm(abs(vec(Array(d))),rev=true)[1:gcheck]
    end
    pass = true
    for i in irange
        w0 = w[i]
        (w1, w2) = gc_interval(w0, delta)
        w[i] = w1
        f1 = f(worig, x...; kwargs...)
        w[i] = w2
        f2 = f(worig, x...; kwargs...)
        w[i] = w0
        nd = (f2-f1) / (w2-w1)
        di = (d==nothing ? zero(nd) : d[i])
        if !isapprox(di, nd; rtol=rtol, atol=atol)
            if verbose; warn("d=$di nd=$nd"); end
            pass = false
        else
            if verbose; println("gcheck: d=$di nd=$nd"); end
        end
    end
    return pass
end

function gc_number(d, f, w, x...; delta=dx(w),rtol=dx(w),atol=dx(w),verbose=false,kwargs=[])
    (w1, w2) = gc_interval(w, delta)
    (f1, f2) = (f(w1,x...;kwargs...), f(w2,x...;kwargs...))
    nd = (f2-f1) / (w2-w1)
    di = (d==nothing ? zero(nd) : d)
    if !isapprox(di, nd; rtol=rtol, atol=atol)
        if verbose; warn("d=$d nd=$nd"); end
        return false
    else
        if verbose; println("gcheck: d=$d nd=$nd"); end
        return true
    end
end

dx(x::Number)=cbrt(eps(x))
dx(x::Array)=cbrt(eps(eltype(x)))
dx(x::KnetArray)=cbrt(eps(eltype(x)))
# gc_params(t)=(a=cbrt(eps(t)); (a,a,a))
gc_indices(w)=eachindex(w)
gc_indices(w::Tuple)=(1:length(w))

function gc_interval(w,d)
    w1=w-d/2
    w2=w+d/2
    (w1 < 0 < w) && (w1=zero(w))
    (w2 > 0 > w) && (w2=zero(w))
    return (w1,w2)
end

function gc_scalar(f)
    r = MersenneTwister()
    function g(x...; o...)
        srand(r,1)
        y = f(x...; o...)
        v = AutoGrad.getval(y)
        a = oftype(v, rand(r, size(v)))
        sum(y .* a)
    end
    return g
end

