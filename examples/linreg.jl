# Simple linear regression.
module LinReg
using Main, Knet, ArgParse

# Main loop:

function main(args=ARGS)
    info("Simple linear regression example")
    @show args
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--inputdims"; arg_type=Int; default=100)
        ("--outputdims"; arg_type=Int; default=10)
        ("--batchsize"; arg_type=Int; default=20)
        ("--epochsize"; arg_type=Int; default=10000)
        ("--epochs"; arg_type=Int; default=5)
        ("--noise"; arg_type=Float64; default=0.01)
        ("--gcheck"; arg_type=Int; default=0)
        ("--lr"; arg_type=Float64; default=0.05)
        ("--seed"; arg_type=Int; default=42)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args,s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    global data = Data(outputdims, inputdims; batchsize=batchsize, epochsize=epochsize, noise=noise)
    global net = compile(:wdot; out=outputdims)
    setp(net; lr=lr)
    losscnt = zeros(2)
    maxnorm = zeros(2)
    for epoch = 1:epochs
        train(net, data, quadloss; maxnorm=fill!(maxnorm,0), losscnt=fill!(losscnt,0))
        println((losscnt[1]/losscnt[2], maxnorm[1], maxnorm[2]))
        gcheck > 0 && gradcheck(net, f->gradloss(f,data,quadloss;grad=true), f->gradloss(f,data,quadloss); gcheck=gcheck)
    end
    return (losscnt[1]/losscnt[2], maxnorm[1], maxnorm[2])
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

function gradloss(f, data, loss; grad=false, seed=42)
    data_rng = data.rng
    data.rng = MersenneTwister()
    srand(data.rng, seed)
    (x,ygold) = first(data)
    ypred = forw(f, x)
    grad && back(f, ygold, loss)
    data.rng = data_rng
    loss(ypred, ygold)
end

# Data generator:
import Base: start, next, done

type Data; w; batchsize; epochsize; noise; rng; end

function Data(outputdims,inputdims; batchsize=20, epochsize=10000, noise=.01, rng=Base.GLOBAL_RNG)
    Data(randn(rng,outputdims,inputdims),batchsize,epochsize,noise,rng)
end

function next(l::Data, n)
    (outputdims, inputdims) = size(l.w)
    x = rand(l.rng, inputdims, l.batchsize)
    y = l.w * x + scale(l.noise, randn(l.rng, outputdims, l.batchsize))
    return ((x,y), n+l.batchsize)
end

start(l::Data)=0
done(l::Data,n)=(n >= l.epochsize)

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end
