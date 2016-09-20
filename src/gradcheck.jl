"""

gradcheck(f, w, x...; gcheck=10, o...)

Numerically checks the gradient of `f(w,x...;o...)` with respect to
its first argument `w`, which could be a Number, Array, Tuple or Dict.
`gcheck` largest entries from each numeric array in gradient
`dw=(grad(f))(w,x...;o...)` is compared to its numerical estimate and
a Warning is issued for comparisons that fail.

"""
function gradcheck(f, w, x...; gcheck=10, o...)
    g = grad(f)
    d = g(w, x...; o...)
    if isa(w, Number)
        gc_number(d, f, w, x...; o...)
    elseif isa(w, KnetArray) || (isa(w, Array) && isbits(eltype(w)))
        gc_array(w, d, f, w, x...; gcheck=gcheck, o...)
    else
        k = gc_indices(w)
        for i in k
            gc_index(w, d, i, f, w, x...; gcheck=gcheck, o...)
        end
    end
end

function gc_index(w, d, i, f, w0, x...; gcheck=10, o...)
    if isa(w[i], Number)
        gc_array(w, d, f, w0, x...; gcheck=1, icheck=i, o...)
    elseif isa(w[i],KnetArray) || (isa(w[i], Array) && isbits(eltype(w[i])))
        gc_array(w[i], d[i], f, w0, x...; gcheck=gcheck, o...)
    else
        k = gc_indices(w[i])
        for j in k
            gc_index(w[i], d[i], j, f, w0, x...; gcheck=gcheck, o...)
        end
    end
end

function gc_array(w, d, f, worig, x...; gcheck=10, icheck=0, o...)
    irange = (icheck > 0 ? (icheck:icheck) :
              length(w) <= gcheck ? (1:length(w)) :
              d == nothing ? rand(1:length(w), gcheck) :
              sortperm(abs(vec(Array(d))),rev=true)[1:gcheck])
    (delta, atol, rtol) = gc_params(typeof(w[first(irange)]))
    for i in irange
        w0 = w[i]
        (w1, w2) = gc_interval(w0, delta)
        w[i] = w1
        f1 = f(worig, x...; o...)
        w[i] = w2
        f2 = f(worig, x...; o...)
        w[i] = w0
        nd = (f2-f1) / (w2-w1)
        di = (d==nothing ? 0 : d[i])
        if !isapprox(di, nd; rtol=rtol, atol=atol)
            warn("d=$di nd=$nd")
        else
            println("gcheck: d=$di nd=$nd")
        end
    end
end

function gc_number(d, f, w, x...; o...)
    (delta, atol, rtol) = gc_params(typeof(w))
    (w1, w2) = gc_interval(w, delta)
    (f1, f2) = (f(w1,x...;o...), f(w2,x...;o...))
    nd = (f2-f1) / (w2-w1)
    if !isapprox(d, nd; rtol=rtol, atol=atol)
        warn("d=$d nd=$nd")
    else
        println("gcheck: d=$d nd=$nd")
    end
end

gc_params(t)=(a=cbrt(eps(t)); (a,a,a))
gc_indices(w)=eachindex(w)
gc_indices(w::Tuple)=(1:length(w))

function gc_interval(w,d)
    w1=w-d/2
    w2=w+d/2
    (w1 < 0 < w) && (w1=zero(w))
    (w2 > 0 > w) && (w2=zero(w))
    return (w1,w2)
end
