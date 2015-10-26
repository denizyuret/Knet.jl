immutable Tagger <: Model; forw; back; pred; params;
    function Tagger(forw::Function, back::Function, pred::Function; o...)
        forw = Net(forw; o...)
        back = Net(back; o...)
        pred = Net(pred; o..., ninputs=2)
        par = vcat(params(forw), params(back), params(pred))
        new(forw, back, pred, par)
    end
end

params(m::Tagger)=m.params
reset!(m::Tagger)=(reset!(m.forw);reset!(m.back);reset!(m.pred))
test(m::Tagger, data, loss; o...)=(l=zeros(2); tagger_loop(m, data, loss; losscnt=l, o...); l[1]/l[2])
train(m::Tagger, data, loss; o...)=tagger_loop(m, data, loss; trn=true, o...) 

function tagger_loop(m::Tagger, data, loss; gcheck=false, o...)                     # t3:7904	
    x,ygold,mask = Any[],Any[],Any[]
    for item in data                                                            # each item contains one (x,y,mask) triple for a minibatch of tokens for time t
        if item != nothing      
            push!(x, copy(item[1])); push!(ygold, copy(item[2])); push!(mask, copy(item[3]))
        else                                                                    # or an item can be nothing marking sentence end
            reset!(m)
            yforw = tagger_forw(m.forw, x; o...)                                # t3:1309
            yback = reverse(tagger_forw(m.back, reverse(x); o...))              # t3:1354
            ypred = tagger_forw(m.pred, yforw, yback; o...)                     # t3:323 pred network should accept two inputs
            tagger_loss(ypred, ygold, mask, loss; o...)
            tagger_bptt(m, ygold, mask, loss; o...)                             # t3:4858
            gcheck && break     # only do one sentence and no update when gradient checking
            tagger_update(m; o...)
            empty!(x); empty!(ygold); empty!(mask)
        end
    end
end

function tagger_forw(net::Net, inputs...; o...)
    N = length(inputs[1])
    ystack = cell(N)
    for n=1:N
        ypred = forw(net, map(x->x[n], inputs)...; seq=true, o...)
        ystack[n] = copy(ypred)
    end
    return ystack
end

function tagger_loss(ypred, ygold, mask, loss; losscnt=nothing, maxnorm=nothing, lossreport=0, o...)
    losscnt==nothing && return
    (yrows, ycols) = size2(ypred[1])
    for i=1:length(ypred)
        @assert (yrows, ycols) == size2(ypred[i])
        ntoks = (mask[i] == nothing ? ycols : sum(mask[i]))
        losscnt[1] += loss(ypred[i], ygold[i]; mask=mask[i], o...)
        losscnt[2] += ntoks/ycols
    end
    if lossreport > 0 && losscnt[2]*ycols > lossreport
        println((exp(losscnt[1]/losscnt[2]), losscnt[1]*ycols, losscnt[2]*ycols, (maxnorm==nothing ? [] : maxnorm)...))
        losscnt[1] = losscnt[2] = 0
        maxnorm == nothing || (maxnorm[1] = maxnorm[2] = 0)
    end
end

function tagger_bptt(m::Tagger, ygold, mask, loss; trn=false, o...)             # t3:4858
    trn || return
    N=length(ygold)
    gforw,gback = cell(N),cell(N)
    for n=N:-1:1
        (gf, gb) = back(m.pred, ygold[n], loss; seq=true, mask=mask[n], getdx=true, o...) # t3:620
        gforw[n],gback[n] = copy(gf),copy(gb)
    end
    for n=1:N
        back(m.back, gback[n]; seq=true, mask=mask[n], o...)                    # t3:2104
    end
    for n=N:-1:1
        back(m.forw, gforw[n]; seq=true, mask=mask[n], o...)                    # t3:2069
    end
end

function tagger_update(m::Tagger; gclip=0, trn=false, maxnorm=nothing, o...)
    trn || return
    g = nothing
    if gclip > 0
        g = gnorm(m)
        gclip=(g > gclip > 0 ? gclip/g : 0)
    end
    update!(m; gclip=gclip, o...)
    if maxnorm != nothing
        w = wnorm(m)
        g == nothing && (g = gnorm(m))
        w > maxnorm[1] && (maxnorm[1]=w)
        g > maxnorm[2] && (maxnorm[2]=g)
    end
end
