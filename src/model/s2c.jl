immutable S2C <: Model; rnn; fnn; params; end

S2C(rnn::Net,fnn::Net)=S2C(rnn,fnn,vcat(params(rnn),params(fnn)))
S2C(rnn::Expr,fnn::Expr)=S2C(Net(rnn),Net(fnn))
params(m::S2C)=m.params

function train(m::S2C, data, loss; getloss=true, getnorm=true, gclip=0, gcheck=0, o...)
    gcheck>0 && Base.warn_once("s2c.gradcheck has not been implemented, ignoring option.")
    numloss = sumloss = maxwnorm = maxgnorm = w = g = 0
    for item in data
        (x,ygold) = item2xy(item)
        ycell = forw(m.rnn, x...; trn=true, seq=true, o...)     # TODO: do we need trn/seq everywhere?
        ygold == nothing && continue                            # ygold!=nothing signals end of sequence
        ypred = forw(m.fnn, ycell; trn=true, o...)              # TODO: avoid unnecessary copy of ycell
        getloss && (sumloss += loss(ypred, ygold); numloss += 1)
        ygrad = back(m.fnn, ygold, loss; getdx=true, o...)
        back(m.rnn, ygrad; seq=true, o...)                      # TODO: avoid unnecessary copy of ygrad
        while m.rnn.sp > 0
            back(m.rnn; seq=true, o...)
        end
        (getnorm || gclip>0) && (g = gnorm(m); g > maxgnorm && (maxgnorm = g))
        update!(m; gclip=(g > gclip > 0 ? gclip/g : 0), o...)
        getnorm && (w = wnorm(m); w > maxwnorm && (maxwnorm = w))
        reset!(m.rnn; o...)
    end
    return (sumloss/numloss, maxwnorm, maxgnorm)
end

function test(m::S2C, data, loss; o...)
    sumloss = numloss = 0
    for item in data
        (x,ygold) = item2xy(item)
        ycell = forw(m.rnn, x...; trn=false, seq=true, o...)
        ygold == nothing && continue
        ypred = forw(m.fnn, ycell; trn=false, o...)
        sumloss += loss(ypred, ygold); numloss += 1
        reset!(m.rnn; o...)
    end
    sumloss / numloss
end
