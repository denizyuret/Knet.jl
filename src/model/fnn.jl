immutable FNN <: Model; net; end

FNN(e::Expr)=FNN(Net(e))
params(m::FNN)=params(m.net)

function train(m::FNN, data, loss; gclip=0, gcheck=0, getnorm=true, getloss=true, o...)
    gcheck > 0 && (gradcheck(m, data, loss; gcheck=gcheck, o...); gcheck=0)
    numloss = sumloss = maxwnorm = maxgnorm = w = g = 0
    for item in data
        (x,ygold) = item2xy(item)
        ypred = forw(m.net, x...; trn=true, o...)
        getloss && (sumloss += loss(ypred, ygold); numloss += 1)
        back(m.net, ygold, loss; o...)
        (getnorm || gclip>0) && (g = gnorm(m); g > maxgnorm && (maxgnorm = g))
        update!(m; gclip=(g > gclip > 0 ? gclip/g : 0), o...)
        getnorm && (w = wnorm(m); w > maxwnorm && (maxwnorm = w))
    end
    return (sumloss/numloss, maxwnorm, maxgnorm)
end

function test(m::FNN, data, loss; o...)
    sumloss = numloss = 0
    for item in data
        (x,ygold) = item2xy(item)
        ypred = forw(m.net, x...; trn=false, o...)
        sumloss += loss(ypred, ygold); numloss += 1
    end
    sumloss / numloss
end

function predict(m::FNN, data; o...)
    y = Any[]
    for x in data
        ypred = forw(m.net, x...; trn=false, o...)
        ycopy = isa(ypred, Array) ? copy(ypred) : convert(Array, ypred)
        push!(y, ycopy)
    end
    return y
end

function gradcheck(m::FNN, data, loss; delta=1e-4, rtol=eps(Float64)^(1/5), atol=eps(Float64)^(1/5), gcheck=10, o...)
    x = ygold = loss0 = nothing
    for item in data
        (x,ygold) = item2xy(item)
        ypred = forw(m.net, x...; trn=true, o...)
        loss0 = loss(ypred, ygold)
        back(m.net, ygold, loss; o...)
        break
    end
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
            ypred = forw(m.net, x...; trn=false, o...)
            loss1 = loss(ypred, ygold)
            p.out[i] = wi0
            dwi = (loss1 - loss0) / (wi1 - wi0)
            if !isapprox(pdiff[i], dwi; rtol=rtol, atol=atol)
                println(tuple(:gc, n, i, pdiff[i], dwi))
            end
        end
        @assert isequal(p.out, psave)
    end
end

