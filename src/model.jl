"""
Model is an abstract type whose subtypes should provide the following:

* `train(model, data, loss)`
* `test(model, data, loss)`
* `predict(model, data)`
* `setopt!(model; param...)`
"""
abstract Model

setopt!(m::Model; o...)=(for p in params(m); setopt!(p; o...); end)
update!(m::Model; o...)=(for p in params(m); update!(p; o...); end)             # t:19
vnorm(x)=(x==nothing ? 0 : vecnorm(x))
wnorm(m::Model,w=0)=(for p in params(m); w += vnorm(p.out); end; w)           # t:317
gnorm(m::Model,g=0)=(for p in params(m); g += vnorm(p.dif); end; g)           # t:332

# Helper function for train/test
item2xy(item)=(isa(item, Tuple) ? (item[1:end-1],item[end]) : item==nothing ? (nothing,nothing) : ((),item))

# So gradient checking does not mess up random seed:
const gradcheck_rng = MersenneTwister()


# train(m, d; seq=false, a...)=(!seq ? train1(m,d;a...) : train2(m,d;a...))

# function train1(m::Model, data; loss=quadloss, gclip=0, gcheck=0, getnorm=true, getloss=true, a...)
#     numloss = sumloss = maxwnorm = maxgnorm = w = g = 0
#     for item in data
#         (x,ygold) = item2xy(item)
#         gcheck > 0 && (gradcheck(m,ygold,x...; gcheck=gcheck, loss=loss, a...); gcheck=0)
#         ypred = forw(m, x...; trn=true, a...)
#         getloss && (sumloss += loss(ypred, ygold); numloss += 1)
#         back(m, ygold; loss=loss, a...)
#         (getnorm || gclip>0) && (g = gnorm(m); g > maxgnorm && (maxgnorm = g))
#         update!(m; gclip=(g > gclip > 0 ? gclip/g : 0), a...)
#         getnorm && (w = wnorm(m); w > maxwnorm && (maxwnorm = w))
#     end
#     return (sumloss/numloss, maxwnorm, maxgnorm)
# end

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

