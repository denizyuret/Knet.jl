"""
Model is an abstract type whose subtypes should provide the following:

* `y=forw(m,x...; trn)`		# returns internal y
* `l=loss(m,y,ygold,ygrad)`	# overwrites ygrad and returns loss
* `back(m,ygrad,dx...)`		# overwrites dx if != nothing
* `params(m)`

Using these low level methods, Model defines the following:

* `train(model, data; gclip, gcheck, getloss, getnorm)`
* `test(model, data)`
* `predict(model, data)` (TODO)
* `accuracy(model, data)`
* `setopt!(model; param...)`
"""
abstract Model

function train(m::Model, data; loss=quadloss, gclip=0, gcheck=0, getnorm=true, getloss=true, seq=false, a...)
    numloss = sumloss = maxwnorm = maxgnorm = w = g = 0
    for item in data # TODO: implement nothing and reset for rnn
        (x,ygold) = item2xy(item)
        ypred = forw(m, x...; trn=true, seq=seq, a...)
        back(m, ygold; loss=loss, seq=seq, a...)

        getnorm && (w = wnorm(m); w > maxwnorm && (maxwnorm = w))
        (getnorm || gclip>0) && (g = gnorm(m); g > maxgnorm && (maxgnorm = g))
        getloss && (sumloss += loss(ypred, ygold); numloss += 1)
        gcheck > 0 && (gradcheck(m,ygold,x...; gcheck=gcheck, loss=loss, seq=seq, a...); gcheck=0) # when to do this in seq

        update!(m; gclip=(g > gclip > 0 ? gclip/g : 0))
    end
    return (sumloss/numloss, maxwnorm, maxgnorm)
end

function test(m::Model, data; loss=quadloss, a...)
    sumloss = numloss = 0
    for item in data
        (x,ygold) = item2xy(item)
        ypred = forw(m, x...; trn=false, seq=false, a...)
        sumloss += loss(ypred, ygold); numloss += 1
    end
    sumloss / numloss
end

setopt!(m::Model; o...)=(for p in params(m); setopt!(p; o...); end)
update!(m::Model; o...)=(for p in params(m); update!(p; o...); end)             # t:19
wnorm(m::Model,w=0)=(for p in params(m); w += vecnorm(p.out); end; w)           # t:317
gnorm(m::Model,g=0)=(for p in params(m); g += vecnorm(p.dif); end; g)           # t:332
item2xy(item)=(isa(item, Tuple) ? (item[1:end-1],item[end]) : item==nothing ? (nothing,nothing) : ((),item))

const gradcheck_rng = MersenneTwister()

function gradcheck(m::Model, ygold, x...; delta=1e-4, rtol=eps(Float64)^(1/5), atol=eps(Float64)^(1/5), gcheck=10, loss=quadloss, seq=false, a...)
    ypred = forw(m, x...; trn=true, seq=seq, a...)
    l0 = loss(ypred, ygold)
    back(m, ygold; loss=loss, seq=seq, a...)
    pp = params(m)
    for n=1:length(pp)
        p = pp[n]
        psave = copy(p.out)
        pdiff = convert(Array, p.dif)
        wlen = length(p.out)
        irange = (wlen <= gcheck ? (1:wlen) : rand(gradcheck_rng, 1:wlen, gcheck))
        for i in irange
            wi0 = p.out[i]
            wi1 = (wi0 >= 0 ? wi0 + delta : wi0 - delta)
            p.out[i] = wi1
            ypred = forw(m, x...; trn=false, seq=seq, a...)
            l1 = loss(ypred, ygold)
            p.out[i] = wi0
            dwi = (l1 - l0) / (wi1 - wi0)
            if !isapprox(pdiff[i], dwi; rtol=rtol, atol=atol)
                println(tuple(:gc, n, i, pdiff[i], dwi))
            end
        end
        @assert isequal(p.out, psave)
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

# TODO: this does not work, cannot write back on data
# function predict(m::Model, d)
#     for (x,y) in d
#         forw(m, x; yout=y, trn=false)
#     end
# end

# function backprop(m::Model, x, y; getloss=true, a...)
#     loss1 = forw(m, x; trn=true, ygold=(getloss ? y : nothing), a...)
#     back(m, y; a...)
#     return loss1
# end

# Use test with percloss instead:
# 
# function accuracy(m::Model, d) # TODO: this only works if y is a single item
#     numcorr = numinst = 0
#     z = nothing
#     for (x,y) in d
#         z == nothing && (z = Array(eltype(y), size(y)))
#         forw(m, x, z; mode=:test)
#         numinst += ccount(y)
#         numcorr += sum(findmax(convert(Array,y),1)[2] .== findmax(convert(Array,z),1)[2])
#     end
#     return numcorr/numinst
# end


# function train1(m::Model, data; loss=quadloss, getloss=true, getnorm=true, gclip=0, gcheck=0, seq=false, a...)
#     numloss = sumloss = maxwnorm = maxgnorm = w = g = 0
#     for item in data
#         (x,ygold) = item2xy(item)
#         ypred = forw(m, x...; trn=true, seq=false, a...)
#         l = getloss ? loss(ypred, ygold) : 0
#         sumloss += l; numloss += 1
#         back(m, ygold; loss=loss, seq=false, a...)
#         getnorm && (w = wnorm(m); w > maxwnorm && (maxwnorm = w))
#         (getnorm || gclip>0) && (g = gnorm(m); g > maxgnorm && (maxgnorm = g))
#         update!(m; gclip=(g > gclip > 0 ? gclip/g : 0))
#     end
#     return (sumloss/numloss, maxwnorm, maxgnorm)
# end

# function test(m::Model, d; o...)
#     sumloss = numloss = 0
#     for (x,y) in d
#         sumloss += forw(m, x; mode=:test, ygold=y, o...)
#         numloss += 1
#     end
#     return sumloss/numloss
# end

