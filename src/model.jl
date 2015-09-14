"""
Model is an abstract type whose subtypes should provide the following:

* `forw(m,x)`
* `back(m,y)`
* `loss(m,y)`
* `update(m)`
* `nops(m)`
* `op(m,n)`

Using these low level methods, Model defines the following:

* `train(m,x,y)`
* `predict(m,x,y)`
* `test(m,x,y)`    
* `gradcheck(m,x,y)`
* `setparam!(m;p...)`
"""
abstract Model

# TODO: do we need to unfold x in a loop?

function backprop(m::Model,x,y; getloss=false)
    forw(m, x; train=true)
    rval = (getloss ? loss(m, y) : nothing)
    back(m, y)
    return rval
end

function train(m::Model, x, y; getloss=false, getnorm=false, gclip=0)
    l = backprop(m,x,y; getloss=getloss)
    (getnorm || gclip>0) && ((w,g)=sumnorm(m))
    g > gclip > 0 && gscale!(m, gclip/g)
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
    dw = cell(nops(m))
    for n=1:length(dw)
        p = param(op(m,n))
        dw[n] = (p == nothing ? nothing : convert(Array, p.diff))
    end
    for n=1:length(dw)
        dw[n] == nothing && continue
        p = param(op(m,n))
        pcopy = copy(p.arr)
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

function sumnorm(m::Model, w=0, g=0)
    for n=1:nops(m)
        p = param(op(m,n))
        p == nothing && continue
        w += vecnorm(p.arr)
        g += vecnorm(p.diff)
    end
    return (w, g)
end

function gscale!(m::Model, s)
    for n=1:nops(m)
        p = param(op(m,n))
        p == nothing && continue
        scale!(s, p.diff)
    end
end

function setparam!(m::Model; o...)
    for n=1:nops(m)
        p = param(op(m,n))
        p == nothing && continue
        setparam!(p; o...)
    end
end
