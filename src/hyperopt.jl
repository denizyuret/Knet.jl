"""

    goldensection(f,n;kwargs) => (fmin,xmin)

Find the minimum of `f` using concurrent golden section search in `n`
dimensions. See `Knet.goldensection_demo()` for an example.

`f` is a function from a `Vector{Float64}` of length `n` to a
`Number`.  It can return `NaN` for out of range inputs.  Goldensection
will always start with a zero vector as the initial input to `f`, and
the initial step size will be 1 in each dimension.  The user should
define `f` to scale and shift this input range into a vector
meaningful for their application. For positive inputs like learning
rate or hidden size, you can use a transformation such as `x0*exp(x)`
where `x` is a value `goldensection` passes to `f` and `x0` is your
initial guess for this value. This will effectively start the search
at `x0`, then move with multiplicative steps.

I designed this algorithm combining ideas from [Golden Section
Search](http://apps.nrbook.com/empanel/index.html?pg=492) and [Hill
Climbing Search](https://en.wikipedia.org/wiki/Hill_climbing). It
essentially runs golden section search concurrently in each dimension,
picking the next step based on estimated gain.

# Keyword arguments    
* `dxmin=0.1`: smallest step size.
* `accel=φ`: acceleration rate. Golden ratio `φ=1.618...` is best.
* `verbose=false`: use `true` to print individual steps.
* `history=[]`: cache of `[(x,f(x)),...]` function evaluations.

"""
function goldensection(f,n; dxmin=0.1, accel=golden, history=[], verbose=false)

    function feval(x)           # so we don't repeat function evaluations
        for (k,v) in history
            if isapprox(x,k)
                return v
            end
        end
        fx = f(x)
        push!(history, (x,fx))
        return fx
    end

    function setindex(x,v,d)    # non-mutating setindex
        y = copy(x)
        y[d] = v
        return y
    end

    x0 = zeros(n)               # initial point
    f0 = feval(x0)              # initial value
    dx = ones(n)                # step sizes
    df = Inf * ones(n)          # expected gains
    while maximum(abs(dx)) >= dxmin
        i = indmax(df)
        x1 = setindex(x0,x0[i]+dx[i],i)
        f1 = feval(x1)
        if verbose; println((:f0,f0,:x0,x0,:f1,f1,:x1,x1,:dx,dx,:df,df)); end
        isnan(f1) && (f1=f0+df[i])
        if f1 < f0
            dx[i] = accel * dx[i]
            df[i] = accel * (f0-f1)
            x0,f0 = x1,f1
            for j = 1:length(df)
                if abs(dx[j]) < dxmin * accel
                    dx[j] = sign(dx[j]) * dxmin * accel
                    df[j] = max(df[j],0) # max(df[j],-1-df[j])
                end
            end
        else
            dx[i] = -dx[i] / accel
            df[i] = (f1-f0) / accel
            if abs(dx[i]) < dxmin
                df[i] = -1 # -1-df[i]
            end
        end
    end
    return (f0,x0)
end

function goldensection_demo()
    include(Knet.dir("examples","mnist.jl"))
    neval = 0

    function f(x)
        neval += 1
        winit,lr,hidden = xform(x)
        if hidden < 10000
            w = MNIST.main("--winit $winit --lr $lr --hidden $hidden --seed 1 --epochs 1 --fast")
            corr,loss = MNIST.accuracy(w,MNIST.dtst)
        else
            corr,loss = NaN,NaN # prevent huge weights
        end
        println((:neval,neval,:loss,loss,:corr,corr,:winit,winit,:lr,lr,:hidden,hidden))
        return loss
    end

    function xform(x)
        winit,lr,hidden = exp(x) .* [ 0.01, 0.01, 100.0 ]
        hidden = ceil(Int, hidden)
        (winit,lr,hidden)
    end

    goldensection(f,3)
end


"""

    hyperband(getconfig, getloss, maxresource=27, reduction=3)

Hyperparameter optimization using the hyperband algorithm from ([Lisha
et al. 2016](https://arxiv.org/abs/1603.06560)).  You can try a simple
MNIST example using `Knet.hyperband_demo()`. 

# Arguments
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

# TODO: document Successive/Sequential Halving:
# http://www.jmlr.org/proceedings/papers/v51/jamieson16.pdf,
# http://www.jmlr.org/proceedings/papers/v28/karnin13.pdf,
# Successive  Reject:
# http://certis.enpc.fr/~audibert/Mes%20articles/COLT10.pdf
function halving(getconfig, getloss, n, r=1, reduction=3, s=round(Int, log(n)/log(reduction)))
    best = (Inf,)
    T = [ getconfig() for i=1:n ]
    for i in 0:s
        ni = floor(Int,n/(reduction^i))
        ri = r*(reduction^i)
        # println((:s,s,:n,n,:r,r,:i,i,:ni,ni,:ri,ri,:T,length(T)))
        L = [ getloss(t, ri) for t in T ]
        l = sortperm(L); l1=l[1]
        L[l1] < best[1] && (best = (L[l1],ri,T[l1])) # ;println("best1: $best"))
        T = T[l[1:floor(Int,ni/reduction)]]
    end
    # println("best2: $best")
    return best
end

function hyperband_demo()
    include(Knet.dir("examples","mnist.jl"))
    best = (Inf,)
    neval = 0

    function getloss(config,epochs)
        neval += 1
        winit,lr,hidden = config
        epochs = round(Int,epochs)
        w = MNIST.main("--winit $winit --lr $lr --hidden $hidden --epochs $epochs --seed 1 --fast")
        corr,loss = MNIST.accuracy(w,MNIST.dtst)
        println((:epochs,epochs,:corr,corr,:loss,loss,:winit,winit,:lr,lr,:hidden,hidden))
        if loss < best[1]
            best = (loss, config, epochs)
        end
        return loss
    end

    function getconfig()
        winit = 0.001^rand()
        lr = 0.001^rand()
        hidden = 16 + floor(Int, 10000^rand())
        return (winit,lr,hidden)
    end

    hyperband(getconfig, getloss)
    println((:neval,neval,:minloss,best[1],:epochs,best[3],:winit,best[2][1],:lr,best[2][2],:hidden,best[2][3]))
end

