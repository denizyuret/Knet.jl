# Simple linear regression.

using Knet, ArgParse
import Base.LinAlg.linreg

# Main loop:

function linreg(args=ARGS)
    info("Simple linear regression example")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--inputs"; arg_type=Int; default=100)
        ("--outputs"; arg_type=Int; default=10)
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
    global data = LinReg(outputs, inputs; batchsize=batchsize, epochsize=epochsize, noise=noise)
    global net = FNN(wdot; out=outputs)
    setopt!(net; lr=lr)
    losscnt = zeros(2)
    maxnorm = zeros(2)
    for epoch = 1:epochs
        train(net, data, quadloss; maxnorm=fill!(maxnorm,0), losscnt=fill!(losscnt,0))
        println((losscnt[1]/losscnt[2], maxnorm[1], maxnorm[2]))
        gcheck > 0 && gradcheck(net, data, quadloss; gcheck=gcheck)
    end
    return (losscnt[1]/losscnt[2], maxnorm[1], maxnorm[2])
end

# Data generator:
import Base: start, next, done

type LinReg; w; batchsize; epochsize; noise; rng; end

function LinReg(outputs,inputs; batchsize=20, epochsize=10000, noise=.01, rng=Base.GLOBAL_RNG)
    LinReg(randn(rng,outputs,inputs),batchsize,epochsize,noise,rng)
end

function next(l::LinReg, n)
    (outputs, inputs) = size(l.w)
    x = rand(l.rng, inputs, l.batchsize)
    y = l.w * x + scale(l.noise, randn(l.rng, outputs, l.batchsize))
    return ((x,y), n+l.batchsize)
end

start(l::LinReg)=0
done(l::LinReg,n)=(n >= l.epochsize)

!isinteractive() && !isdefined(:load_only) && linreg(ARGS)
