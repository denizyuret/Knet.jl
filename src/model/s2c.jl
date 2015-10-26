immutable S2C <: Model; rnn; fnn; params;
    function S2C(rnn::Function,fnn::Function; o...)
        r = Net(rnn; o...)
        f = Net(fnn; o...)
        p = vcat(params(r), params(f))
        new(r, f, p)
    end
end

params(m::S2C)=m.params
reset!(m::S2C;o...)=(reset!(m.rnn;o...); reset!(m.fnn;o...))

train(m::S2C, data, loss; o...)=s2c_loop(m, data, loss; trn=true, o...)
test(m::S2C, data, loss; o...)=(l=zeros(2); s2c_loop(m, data, loss; losscnt=l, o...); l[1]/l[2])

function s2c_loop(m::S2C, data, loss; gcheck=false, o...)
    reset!(m; o...)
    for item in data
        (x,ygold) = item2xy(item)
        ycell = forw(m.rnn, x...; seq=true, o...)
        ygold == nothing && continue
        ypred = forw(m.fnn, ycell; o...)
        s2c_eos(m, x, ygold, ypred, loss; gcheck=gcheck, o...)
        gcheck && break
        reset!(m; o...)
    end
end

function s2c_eos(m::S2C, x, ygold, ypred, loss; trn=false, gcheck=false, maxnorm=nothing, losscnt=nothing, gclip=0, o...)
    if losscnt != nothing
        losscnt[1] += loss(ypred, ygold)
        losscnt[2] += 1
    end
    if trn
        ygrad = back(m.fnn, ygold, loss; getdx=true, o...)
        back(m.rnn, ygrad; seq=true, o...)                      # TODO: avoid unnecessary copy of ygrad
        while m.rnn.sp > 0
            back(m.rnn; seq=true, o...)
        end
        g = (gclip > 0 || maxnorm!=nothing ? gnorm(m) : 0)
        if !gcheck
            gclip=(g > gclip > 0 ? gclip/g : 0)
            update!(m; gclip=gclip, o...)
        end
    end
    if maxnorm != nothing
        w=wnorm(m)
        w > maxnorm[1] && (maxnorm[1]=w)
        g > maxnorm[2] && (maxnorm[2]=g)
    end
end



### DEAD CODE

# Decided not to expose the Net() function.  Users only interact with
# knet functions and models.  Having an intermediate Net object is
# confusing.  To solve the problem of different initialization
# parameters, one could just define new knet functions.

# S2C(rnn::Net,fnn::Net)=S2C(rnn,fnn,vcat(params(rnn),params(fnn)))

# function train(m::S2C, data, loss; getloss=true, getnorm=true, gclip=0, o...)
#     numloss = sumloss = maxwnorm = maxgnorm = w = g = 0
#     for item in data
#         (x,ygold) = item2xy(item)
#         ycell = forw(m.rnn, x...; trn=true, seq=true, o...)     # TODO: do we need trn/seq everywhere?
#         ygold == nothing && continue                            # ygold!=nothing signals end of sequence
#         ypred = forw(m.fnn, ycell; trn=true, o...)              # TODO: avoid unnecessary copy of ycell
#         getloss && (sumloss += loss(ypred, ygold); numloss += 1)
#         ygrad = back(m.fnn, ygold, loss; getdx=true, o...)
#         back(m.rnn, ygrad; seq=true, o...)                      # TODO: avoid unnecessary copy of ygrad
#         while m.rnn.sp > 0
#             back(m.rnn; seq=true, o...)
#         end
#         (getnorm || gclip>0) && (g = gnorm(m); g > maxgnorm && (maxgnorm = g))
#         update!(m; gclip=(g > gclip > 0 ? gclip/g : 0), o...)
#         getnorm && (w = wnorm(m); w > maxwnorm && (maxwnorm = w))
#         reset!(m.rnn; o...)
#     end
#     return (sumloss/numloss, maxwnorm, maxgnorm)
# end

# function test(m::S2C, data, loss; o...)
#     sumloss = numloss = 0
#     for item in data
#         (x,ygold) = item2xy(item)
#         ycell = forw(m.rnn, x...; trn=false, seq=true, o...)
#         ygold == nothing && continue
#         ypred = forw(m.fnn, ycell; trn=false, o...)
#         sumloss += loss(ypred, ygold); numloss += 1
#         reset!(m.rnn; o...)
#     end
#     sumloss / numloss
# end
