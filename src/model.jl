"""
Model is an abstract type whose subtypes should provide the following:

* `forw(m,x)`
* `back(m,y)`
* `loss(m,y)`
* `params(m)`

Using these low level methods, Model defines the following:

* `train(model,data)`
* `test(model,data)`
* `predict(model,data)`
* `setparam!(model; o...)`
"""
abstract Model

setparam!(m::Model; o...)=(for p in params(m); setparam!(p; o...); end)
update(m::Model; o...)=(for p in params(m); update(p; o...); end) # TODO: rename to update!
gscale!(m::Model, s)=(for p in params(m); scale!(s, p.diff); end)
wnorm(m::Model,w=0)=(for p in params(m); w += vecnorm(p.arr); end; w)
gnorm(m::Model,g=0)=(for p in params(m); g += vecnorm(p.diff); end; g)

function backprop(m::Model, x, y; getloss=true)
    forw(m, x; train=true)
    loss1 = getloss ? loss(m, y) : nothing
    back(m, y)
    return loss1
end

function train(m::Model, d::Data; gclip=0, gcheck=0, getloss=true, getnorm=true)
    sumloss = maxwnorm = maxgnorm = w = g = 0
    for (x,y) in d
        gcheck > 0 && (gradcheck(m,x,y; gcheck=gcheck); gcheck=0)
        l = backprop(m,x,y; getloss=getloss)
        getloss && (sumloss += l)
        getnorm && (w = wnorm(m); w > maxwnorm && (maxwnorm = w))
        (getnorm || gclip>0) && (g = gnorm(m); g > maxgnorm && (maxgnorm = g))
        update(m; gclip=(g > gclip > 0 ? gclip/g : 0))
    end
    return (sumloss, maxwnorm, maxgnorm)
end

function test(m::Model, d::Data)
    sumloss = 0
    for (x,y) in d
        forw(m, x; train=false)
        sumloss += loss(m, y)
    end
    return sumloss
end

function predict(m::Model, d::Data)
    for (x,y) in d
        forw(m, x; y=y, train=false)
    end
end

function gradcheck(m::Model, x, y; delta=1e-4, rtol=eps(Float64)^(1/5), atol=eps(Float64)^(1/5), gcheck=10)
    l0 = backprop(m, x, y; getloss=true)
    pp = params(m)
    dw = map(p->convert(Array,p.diff), pp)
    for n=1:length(dw)
        p = pp[n]
        pcopy = copy(p.arr)     # TODO: do we need pcopy?
        w = convert(Array, p.arr)
        wlen = length(w)
        irange = (wlen <= gcheck ? (1:wlen) : rand(1:wlen, gcheck))
        for i in irange
            wi0 = w[i]; wi1 = (wi0 >= 0 ? wi0 + delta : wi0 - delta)
            w[i] = wi1; copy!(p.arr, w); w[i] = wi0
            l1 = test(m,x,y)
            dwi = (l1 - l0) / (wi1 - wi0)
            if !isapprox(dw[n][i], dwi; rtol=rtol, atol=atol)
                println(tuple(:gc, n, i, dw[n][i], dwi))
            end
        end
        copy!(p.arr, pcopy)         # make sure we recover the original
    end
end

# This will not work for MLP!  extra parameterless ops do not effect equality.
# Base.isequal(a::Model,b::Model)=(typeof(a)==typeof(b) && isequal(params(a),params(b)))
