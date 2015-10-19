immutable RNN <: Model; net; RNN(a...;o...)=new(Net(a...;o...)); end
params(m::RNN)=params(m.net)

function train(m::RNN, data, loss; gclip=0, gcheck=false, maxnorm=nothing, losscnt=nothing, o...)
    reset!(m.net; o...)
    ystack = Any[]
    for item in data
        if item != nothing
            (x,ygold) = item2xy(item)
            ypred = forw(m.net, x...; trn=true, seq=true, o...)
            losscnt != nothing && (losscnt[1] += loss(ypred, ygold); losscnt[2] += 1)
            push!(ystack, copy(ygold))
        else                    # end of sequence
            while !isempty(ystack)
                ygold = pop!(ystack)
                back(m.net, ygold, loss; seq=true, o...)
            end
            gcheck && break
            g = (gclip > 0 || maxnorm!=nothing ? gnorm(m) : 0)
            update!(m; gclip=(g > gclip > 0 ? gclip/g : 0), o...)
            if maxnorm != nothing
                w=wnorm(m)
                w > maxnorm[1] && (maxnorm[1]=w)
                g > maxnorm[2] && (maxnorm[2]=g)
            end
            reset!(m.net; o...)
        end
    end
end

function test(m::RNN, data, loss; o...)
    sumloss = numloss = 0
    reset!(m.net; o...)
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

