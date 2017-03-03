using Knet
Atype=(gpu() >= 0 ? KnetArray{Float32} : Array{Float32})

"""

    hyperband(getconfig, getloss, maxresource=27, reduction=3)

Hyperparameter optimization using the hyperband algorithm from ([Lisha et al. 2016](https://arxiv.org/abs/1603.06560)).
You can try a simple MNIST example using `hyperband(getconfig1,getloss1)` after loading this example.

## Arguments
* `getconfig()` returns random configurations with a user defined type and distribution.
* `getloss(c,n)` returns loss for configuration `c` and number of resources (e.g. epochs) `n`.
* `maxresource` is the maximum number of resources any one configuration should be given.
* `reduction` is an algorithm parameter (see paper), 3 is a good value.

"""
function hyperband(getconfig, getloss, maxresource=27, reduction=3)
    smax = floor(Int, log(maxresource)/log(reduction))
    B = (smax + 1) * maxresource
    best = (Inf,)
    for s in smax:-1:0
        n = ceil(Int, (B/maxresource)*((reduction^s)/(s+1)))
        r = maxresource / (reduction^s)
        curr = halving(getconfig, getloss, n, r, reduction, s)
        if curr[1] < best[1]; (best=curr); end
    end
    return best
end

function halving(getconfig, getloss, n, r=1, reduction=3, s=round(Int, log(n)/log(reduction)))
    best = (Inf,)
    T = [ getconfig() for i=1:n ]
    for i in 0:s
        ni = floor(Int,n/(reduction^i))
        ri = r*(reduction^i)
        println((:s,s,:n,n,:r,r,:i,i,:ni,ni,:ri,ri,:T,length(T)))
        L = [ getloss(t, ri) for t in T ]
        l = sortperm(L); l1=l[1]
        L[l1] < best[1] && (best = (L[l1],ri,T[l1]); println("best1: $best"))
        T = T[l[1:floor(Int,ni/reduction)]]
    end
    println("best2: $best")
    return best
end

# An example getconfig and getloss pair that can be used with hyperband:
# Usage: hyperband(getconfig1,getloss1)

function getconfig1()
    pmax = 0.5
    hmin,hmax = 32,512
    pdrop = (rand() < 1/3 ? (rand()*pmax,rand()*pmax) :
             rand() < 1/2 ? (rand()*pmax, 0) : (0, rand()*pmax))
    h = round(Int, hmin+(hmax-hmin)^rand())
    nlayer = rand() < 2/3 ? 1 : rand() < 2/3 ? 2 : 3
    hidden = ntuple(x->h, nlayer)
    return (hidden, pdrop)
end

function getloss1(c,n)
    w = winit(c[1]...)
    p = [ Adam() for wi in w ]
    (epoch,ltrn,ltst) = train(w,p,dtrn,dtst,mlp; epochs=n, pdrop=c[2])
    println((:n,n,:e,epoch,:ltrn,ltrn,:ltst,ltst,:c,c))
    return ltst
end

# Train for `epochs` epochs and return information about the epoch when test loss was best

function train(w,p,dtrn,dtst,model; epochs=100, loss=avgloss, o...)
    best = (0,deepcopy(w),loss(w,dtst,model))
    for epoch in 1:epochs
        for (x,y) in dtrn
            g = softgrad(w,x,y,model; o...)
            update!(w,g,p)
        end
        ltst = loss(w,dtst,model)
        if ltst < best[3]
            best = (epoch,deepcopy(w),ltst)
        end
    end
    (epoch,wbest,ltst) = best
    ltrn = loss(wbest,dtrn,model)
    return (epoch,ltrn,ltst)
end

function winit(h...; atype=Atype, std=0.01, seed=1)
    r = MersenneTwister(seed)
    h = [784, h..., 10]
    w = Any[]
    for i=1:length(h)-1
        push!(w, std*randn(r,h[i+1],h[i]))
        push!(w, zeros(h[i+1],1))
    end
    map(atype, w)
end

function softloss(w,x,p,model;l1=0,l2=0,o...)
    y = model(w,x;o...)
    y = y .- maximum(y,1) # for numerical stability
    expy = exp(y)
    logphat = y .- log(sum(expy,1))
    J = -sum(p .* logphat) / size(x,2)
    if l1 != 0; J += l1 * sum(sumabs(wi)  for wi in w[1:2:end]); end
    if l2 != 0; J += l2 * sum(sumabs2(wi) for wi in w[1:2:end]); end
    return J
end

softgrad = grad(softloss)

function linear(w,x)
    w[1]*x .+ w[2]
end

function dropout(x,p)
    if p == 0
        x
    else
        x .* (rand!(similar(x)) .> p) ./ (1-p)
    end
end

function mlp(w,x; pdrop=(0,0))
    x = dropout(x,pdrop[1])
    for i=1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
        x = dropout(x,pdrop[2])
    end
    return w[end-1]*x .+ w[end]
end

function avgloss(w,data,model)
    sum = cnt = 0
    for (x,y) in data
        sum += softloss(w,x,y,model)
        cnt += 1
    end
    return sum/cnt
end

function zeroone(w,data,model)
    ncorr = ninst = 0
    for (x,y) in data
        ypred = model(w,x)
        ncorr += sum(y .* (ypred .== maximum(ypred,1)))
        ninst += size(x,2)
    end
    return 1 - ncorr/ninst
end

function load_mnist()
    include(Knet.dir("examples","mnist.jl"))
    MNIST.loaddata()
    using MNIST: xtrn,ytrn,xtst,ytst,minibatch
    dtst = minibatch(xtst,ytst,100;atype=Atype)
    dtrn = minibatch(xtrn,ytrn,100;atype=Atype)
end

# For a winit(64) MLP:
# best winit=0.01 tst=.0894 trn=.0422 epoch=13
# best l1=4e-5 tst=.0785 trn=.0408 epoch=26
# best l2=6e-5 tst=.0825 trn=.0293 epoch=26
