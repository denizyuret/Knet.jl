"""
Model is an abstract type whose subtypes should provide the following:

* `forw(m,x)`
* `back(m,y)`
* `loss(m,y)`
* `params(m)`

Using these low level methods, Model defines the following:

* `train(m,x,y)`
* `test(m,x,y)`    
* `predict(m,x,y)`
* `gradcheck(m,x,y)`

and extends the following to apply to each parameter of the model:

* `setparam!(m; o...)`
* `update(m; o...)`
"""
abstract Model

setparam!(m::Model; o...)=(for p in params(m); setparam!(p; o...); end)
update(m::Model; o...)=(for p in params(m); update(p; o...); end) # TODO: rename to update!
gscale!(m::Model, s)=(for p in params(m); scale!(s, p.diff); end)
wnorm(m::Model,w=0)=(for p in params(m); w += vecnorm(p.arr); end; w)
gnorm(m::Model,g=0)=(for p in params(m); g += vecnorm(p.diff); end; g)

# TODO: do we need to unfold x in a loop?

function backprop(m::Model,x,y; getloss=false)
    forw(m, x; train=true)
    rval = (getloss ? loss(m, y) : nothing)
    back(m, y)
    return rval
end

function train(m::Model, x, y; getloss=false, getnorm=false, gclip=0)
    l = backprop(m,x,y; getloss=getloss)
    (getnorm || gclip>0) && (w=wnorm(m); g=gnorm(m))
    g > gclip > 0 && gscale!(m, gclip/g) # TODO: make this an update option
    update(m)
    return (getloss && getnorm ? (l,w,g) : getloss ? l : getnorm ? (w,g) : nothing)
end

function test(m::Model, x, y)
    forw(m, x; train=false)
    loss(m, y)
end

function predict(m::Model, x, y)
    forw(m, x; y=y, train=false)
end

function gradcheck(m::Model, x, y; delta=1e-4, rtol=eps(Float64)^(1/5), atol=eps(Float64)^(1/5), ncheck=10)
    l0 = backprop(m, x, y; getloss=true)
    pp = params(m)
    dw = map(p->convert(Array,p.diff), pp)
    for n=1:length(dw)
        p = pp[n]
        pcopy = copy(p.arr)     # TODO: do we need pcopy?
        w = convert(Array, p.arr)
        wlen = length(w)
        irange = (wlen <= ncheck ? (1:wlen) : rand(1:wlen, ncheck))
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
