immutable RNN <: Model; net; end

RNN(e::Expr)=RNN(Net(e))
params(m::RNN)=params(m.net)

function train(m::RNN, data, loss; gclip=0, gcheck=0, getnorm=true, getloss=true, o...)
    gcheck == 0 || warn("rnn.gradcheck not implemented yet, ignoring option.")
    numloss = sumloss = maxwnorm = maxgnorm = w = g = 0
    ystack = Any[]
    for item in data
        if item != nothing
            (x,ygold) = item2xy(item)
            ypred = forw(m.net, x...; trn=true, seq=true, o...)
            getloss && (sumloss += loss(ypred, ygold); numloss += 1)
            push!(ystack, copy(ygold))
        else                    # end of sequence
            while !isempty(ystack)
                ygold = pop!(ystack)
                back(m.net, ygold, loss; seq=true, o...)
            end
            (getnorm || gclip>0) && (g = gnorm(m); g > maxgnorm && (maxgnorm = g))
            update!(m; gclip=(g > gclip > 0 ? gclip/g : 0), o...)
            getnorm && (w = wnorm(m); w > maxwnorm && (maxwnorm = w))
            reset!(m.net; o...)
        end
    end
    return (sumloss/numloss, maxwnorm, maxgnorm)
end

function test(m::RNN, data, loss; o...)
    sumloss = numloss = 0
    for item in data
        if item != nothing
            (x,ygold) = item2xy(item)
            ypred = forw(m.net, x...; trn=false, seq=true, o...)
            sumloss += loss(ypred, ygold); numloss += 1
        else
            reset!(m.net; o...)
        end
    end
    return sumloss/numloss
end

