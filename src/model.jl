"""
Model is an abstract type whose subtypes should provide the following:

* `forw(m,x;yout,ygold,trn)`
* `back(m,y)`
* `params(m)`

Using these low level methods, Model defines the following:

* `train(model, data; gclip, gcheck, getloss, getnorm)`
* `test(model, data)`
* `predict(model, data)`
* `accuracy(model, data)`
* `setparam!(model; param...)`
"""
abstract Model

setparam!(m::Model; o...)=(for p in params(m); setparam!(p; o...); end)
update!(m::Model; o...)=(for p in params(m); update!(p; o...); end)
wnorm(m::Model,w=0)=(for p in params(m); w += vecnorm(p.arr); end; w)
gnorm(m::Model,g=0)=(for p in params(m); g += vecnorm(p.diff); end; g)

# TODO: this does not work, cannot write back on data
# function predict(m::Model, d)
#     for (x,y) in d
#         forw(m, x; y=y, trn=false)
#     end
# end

function test(m::Model, d; o...)
    sumloss = numloss = 0
    for (x,y) in d
        sumloss += forw(m, x; trn=false, ygold=y, o...)
        numloss += 1
    end
    return sumloss/numloss
end

function accuracy(m::Model, d) # TODO: this only works if y is a single item
    numcorr = numinst = 0
    z = nothing
    for (x,y) in d
        issimilar(y,z) || (z = similar(y))
        forw(m, x; trn=false, yout=z)
        numinst += ccount(y)
        numcorr += sum(findmax(convert(Array,y),1)[2] .== findmax(convert(Array,z),1)[2])
    end
    return numcorr/numinst
end

function train(m::Model, d; gclip=0, gcheck=0, getloss=true, getnorm=true, a...) # TODO: (minor) this should probably be named train!
    numloss = sumloss = maxwnorm = maxgnorm = w = g = 0
    for (x,y) in d
        gcheck > 0 && (gradcheck(m,x,y; gcheck=gcheck, a...); gcheck=0)
        l = backprop(m,x,y; getloss=getloss, a...)
        getloss && (sumloss += l; numloss += 1)
        getnorm && (w = wnorm(m); w > maxwnorm && (maxwnorm = w))
        (getnorm || gclip>0) && (g = gnorm(m); g > maxgnorm && (maxgnorm = g))
        update!(m; gclip=(g > gclip > 0 ? gclip/g : 0))
    end
    return (sumloss/numloss, maxwnorm, maxgnorm)
end

function backprop(m::Model, x, y; getloss=true, a...)
    loss1 = forw(m, x; trn=true, ygold=(getloss ? y : nothing), a...)
    back(m, y; a...)
    return loss1
end

const gradcheck_rng = MersenneTwister()

function gradcheck(m::Model, x, y; delta=1e-4, rtol=eps(Float64)^(1/5), atol=eps(Float64)^(1/5), gcheck=10, a...)
    l0 = backprop(m, x, y; getloss=true)
    pp = params(m)
    dw = map(p->convert(Array,p.diff), pp)
    for n=1:length(dw)
        p = pp[n]
        pcopy = copy(p.arr)     # TODO: do we need pcopy?
        w = convert(Array, p.arr)
        wlen = length(w)
        irange = (wlen <= gcheck ? (1:wlen) : rand(gradcheck_rng, 1:wlen, gcheck))
        for i in irange
            wi0 = w[i]; wi1 = (wi0 >= 0 ? wi0 + delta : wi0 - delta)
            w[i] = wi1; copy!(p.arr, w); w[i] = wi0
            l1 = forw(m, x; trn=false, ygold=y)
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

# function inittrain(m::Model, d)
#     isempty(params(m)[1]) || return
#     (x,n) = next(d,start(d))
#     init(m, x[1]; trn=true)
# end

# NO: make the model interface more functional:
# back and loss rely on hidden state info.  
# forw has to allocate.
# purely functional models are impossible.
# forw needs to compute intermediate values.
# but from user's perspective forw is functional.
# loss/back is not: relying on history.
# we could give them x/y but they would still need internal state.
# if they are going to use internal state they may as well use the one set by forw.
