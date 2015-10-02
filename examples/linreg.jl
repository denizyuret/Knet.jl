# Simple linear regression.

using Knet, ArgParse

# Data generator:
import Base: start, next, done

type LinReg; w; batchsize; epochsize; noise; end

function LinReg(outputs,inputs,batchsize,epochsize,noise)
    LinReg(randn(outputs,inputs),batchsize,epochsize,noise)
end

function next(l::LinReg, n)
    (outputs, inputs) = size(l.w)
    x = rand(inputs, l.batchsize)
    y = l.w * x + scale(l.noise, randn(outputs, l.batchsize))
    return ((x,y), n+l.batchsize)
end

start(::LinReg)=0
done(l::LinReg,n)=(n >= l.epochsize)

# Linear regression model:

LinRegModel(n) = quote
    x = input()
    w = par($n,0)
    y = dot(w,x)
    z = quadloss(y)
end

# Main loop:

function linreg(args=ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--inputs"; arg_type=Int; default=100)
        ("--outputs"; arg_type=Int; default=10)
        ("--batchsize"; arg_type=Int; default=20)
        ("--epochsize"; arg_type=Int; default=10000)
        ("--epochs"; arg_type=Int; default=5)
        ("--noise"; arg_type=Real; default=0.01)
        ("--lr"; arg_type=Real; default=0.05)
        ("--seed"; arg_type=Int; default=42)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args,s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    data = LinReg(outputs, inputs, batchsize, epochsize, noise)
    net = Net(LinRegModel(outputs))
    setopt!(net; lr=lr)
    lwg = nothing
    for epoch = 1:epochs
        @show lwg = train(net, data)
    end
    return lwg
end

!isinteractive() && !isdefined(:load_only) && linreg(ARGS)
