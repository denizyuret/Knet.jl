# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.

isdefined(:MNIST) || include("mnist.jl")

module MNIST2D
using Main,Knet,ArgParse

function main(args=ARGS)
    info("Testing simple mlp on MNIST")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--batchsize"; arg_type=Int; default=100)
        ("--epochs"; arg_type=Int; default=3)
        ("--gcheck"; arg_type=Int; default=0)
        ("--xsparse"; action=:store_true)
        ("--ysparse"; action=:store_true)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args,s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)

    fx = (xsparse ? sparse : identity)
    fy = (ysparse ? sparse : identity)
    global dtrn = minibatch(fx(MNIST.xtrn), fy(MNIST.ytrn), batchsize)
    global dtst = minibatch(fx(MNIST.xtst), fy(MNIST.ytst), batchsize)

    global net = compile(:mnist2layer)
    setopt!(net, lr=0.5)
    # To make sure the array sharing does not change the results:
    # set!(net, :forwoverwrite, false)
    # set!(net, :backoverwrite, false)

    # println((:epoch,:ltrn,:atrn,:ltst,:atst))
    # ltrn = atrn = ltst = atst = 0
    l=[0f0,0f0]; m=[0f0,0f0]

    for epoch=1:epochs
        # train0(net, dtrn, softloss)
        train(net, dtrn, softloss; losscnt=fill!(l,0), maxnorm=fill!(m,0))
        ltrn = test(net, dtrn, softloss)
        atrn = 1-test(net, dtrn, zeroone)
        ltst = test(net, dtst, softloss)
        atst = 1-test(net, dtst, zeroone)
        println((epoch,l[1]/l[2],m[1],m[2],ltrn,atrn,ltst,atst))
        gcheck > 0 && gradcheck(net, f->getgrad(f,dtrn,softloss), f->getloss(f,dtrn,softloss); gcheck=gcheck)
        # println((epoch,ltrn,atrn,ltst,atst))
    end
    # return (epochs,ltrn,atrn,ltst,atst)
    return (l[1]/l[2],m[1],m[2])
end

@knet function mnist2layer(x; winit=Gaussian(0,.1))
    h    = wbf(x; out=64, f=:relu, winit=winit)
    return wbf(h; out=10, f=:soft, winit=winit)
end

function train0(f, data, loss)
    for (x,y) in data
        forw(f, x)
        back(f, y, loss)
        update!(f)
    end
end

function train(f, data, loss; losscnt=nothing, maxnorm=nothing)
    for (x,ygold) in data
        ypred = forw(f, x)
        back(f, ygold, loss)
        update!(f)
        losscnt[1] += loss(ypred, ygold); losscnt[2] += 1
        w=wnorm(f); w > maxnorm[1] && (maxnorm[1]=w)
        g=gnorm(f); g > maxnorm[2] && (maxnorm[2]=g)
    end
end

function test(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in data
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    sumloss / numloss
end

function minibatch(x, y, batchsize)
    data = Any[]
    for i=1:batchsize:size(x,2)-batchsize+1
        j=i+batchsize-1
        push!(data, (x[:,i:j], y[:,i:j]))
    end
    return data
end

function getgrad(f, data, loss)
    (x,ygold) = first(data)
    reset!(f)
    ypred = forw(f, x)
    back(f, ygold, loss)
    loss(ypred, ygold)
end

function getloss(f, data, loss)
    (x,ygold) = first(data)
    reset!(f)
    ypred = forw(f, x)
    loss(ypred, ygold)
end

# This allows both non-interactive (shell command) and interactive calls like:
# julia> mnist2d("--epochs 10")
!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end # module

### SAMPLE RUN

# (epoch,l,w,g,ltrn,atrn,ltst,atst) = (1,0.37387532f0,18.511799f0,2.843379f0,0.21288027f0,0.9327666666666667,0.2148458f0,0.9289000000000001)
# (epoch,l,w,g,ltrn,atrn,ltst,atst) = (2,0.14995994f0,22.269361f0,3.9932733f0,0.13567321f0,0.9574,0.14322147f0,0.9546)
# (epoch,l,w,g,ltrn,atrn,ltst,atst) = (3,0.10628127f0,24.865438f0,3.5134742f0,0.100041345f0,0.9681833333333334,0.114785746f0,0.9641000000000001)


### SAMPLE RUN OLD

# INFO: Loading MNIST...
#   5.736248 seconds (362.24 k allocations: 502.003 MB, 1.35% gc time)
# INFO: Testing simple mlp
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.001) = (0.37387532f0,18.511799f0,2.8433793f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.21288027f0,0.9327666666666666)
# (test(net,dtst),accuracy(net,dtst)) = (0.2148458f0,0.9289)
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.001) = (0.14995994f0,22.26936f0,3.9932733f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.13567321f0,0.9574)
# (test(net,dtst),accuracy(net,dtst)) = (0.14322147f0,0.9546)
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.001) = (0.10628127f0,24.865437f0,3.5134742f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.100041345f0,0.9681833333333333)
# (test(net,dtst),accuracy(net,dtst)) = (0.114785746f0,0.9641)
#  10.373502 seconds (11.14 M allocations: 557.680 MB, 1.25% gc time)
