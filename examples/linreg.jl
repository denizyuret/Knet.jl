# Simple linear regression.
module LinReg
using Knet, AutoGrad, ArgParse

# Main loop:

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="linreg.jl (c) Deniz Yuret, 2016. Linear regression example with artificial data."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--inputdims"; arg_type=Int; default=100)
        ("--outputdims"; arg_type=Int; default=10)
        ("--batchsize"; arg_type=Int; default=20)
        ("--epochsize"; arg_type=Int; default=10000)
        ("--epochs"; arg_type=Int; default=10)
        ("--noise"; arg_type=Float64; default=0.01)
        #TODO ("--gcheck"; arg_type=Int; default=0)
        ("--lr"; arg_type=Float64; default=0.02)
        ("--seed"; arg_type=Int; default=42)
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args,s; as_symbols=true); println(o)
    o[:seed] > 0 && srand(o[:seed])
    data = Data(o[:outputdims], o[:inputdims]; batchsize=o[:batchsize], epochsize=o[:epochsize], noise=o[:noise])
    w = KnetArray(0.1*randn(o[:outputdims], o[:inputdims]))
    println((:epoch,0,:loss,test(w,data)))
    @time w = train(w, data; epochs=o[:epochs], lr=o[:lr])
    println((:epoch,o[:epochs],:loss,test(w,data)))
end

predict(w,x)=(w*x)

loss(w,x,y)=(sum(abs2(y-predict(w,x))) / size(x,2))

lossgradient = grad(loss)

function train(w, data; lr=.02, epochs=10)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            w -= lr * g
        end
        println((:epoch,epoch,:loss,test(w,data)))
    end
    return w
end

function test(w, data)
    sumloss = numloss = 0
    for (x,y) in data
        sumloss += loss(w,x,y)
        numloss += 1
    end
    return sumloss/numloss
end

# Data generator:
import Base: start, next, done

type Data; w; batchsize; epochsize; noise; rng; end

function Data(outputdims,inputdims; batchsize=20, epochsize=10000, noise=.01, rng=Base.GLOBAL_RNG)
    Data(KnetArray(randn(rng,outputdims,inputdims)),batchsize,epochsize,noise,rng)
end

function next(l::Data, n)
    (outputdims, inputdims) = size(l.w)
    x = KnetArray(rand(l.rng, inputdims, l.batchsize))
    y = l.w * x + KnetArray(scale(l.noise, randn(l.rng, outputdims, l.batchsize)))
    return ((x,y), n+l.batchsize)
end

start(l::Data)=0
done(l::Data,n)=(n >= l.epochsize)

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end
