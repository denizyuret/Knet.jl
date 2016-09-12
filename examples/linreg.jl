# Simple linear regression.
module LinReg
using Knet, ArgParse

# Main loop:

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="linreg.jl (c) Deniz Yuret, 2016. Linear regression example with artificial data."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0 ? "KnetArray" : "Array"); help="array type: Array for cpu, KnetArray for gpu")
        ("--batchsize"; arg_type=Int; default=20; help="number of instances in a minibatch")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--epochsize"; arg_type=Int; default=10000; help="number of instances per epoch")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--inputdims"; arg_type=Int; default=100; help="input dimensions")
        ("--lr"; arg_type=Float64; default=0.02; help="learning rate")
        ("--noise"; arg_type=Float64; default=0.01; help="noise in data")
        ("--outputdims"; arg_type=Int; default=10; help="output dimensions")
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args,s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    data = Data(o[:outputdims], o[:inputdims]; batchsize=o[:batchsize], epochsize=o[:epochsize], noise=o[:noise], atype=atype)
    w = convert(atype, 0.1*randn(o[:outputdims], o[:inputdims]))
    println((:epoch,0,:loss,test(w,data)))
    if o[:fast]
        @time w = train(w, data; epochs=o[:epochs], lr=o[:lr])
        println((:epoch,o[:epochs],:loss,test(w,data)))
    else
        @time for epoch=1:o[:epochs]
            w = train(w, data; epochs=1, lr=o[:lr])
            println((:epoch,epoch,:loss,test(w,data)))
            if o[:gcheck] > 0
                gradcheck(loss, w, first(data)...; gcheck=o[:gcheck])
            end
        end
    end
    return w
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

type Data; w; batchsize; epochsize; noise; rng; atype; end

function Data(outputdims,inputdims; batchsize=20, epochsize=10000, noise=.01, rng=Base.GLOBAL_RNG, atype=Array)
    Data(convert(atype, randn(rng,outputdims,inputdims)),batchsize,epochsize,noise,rng,atype)
end

function next(l::Data, n)
    (outputdims, inputdims) = size(l.w)
    x = convert(l.atype, rand(l.rng, inputdims, l.batchsize))
    y = l.w * x + convert(l.atype, scale(l.noise, randn(l.rng, outputdims, l.batchsize)))
    return ((x,y), n+l.batchsize)
end

start(l::Data)=0
done(l::Data,n)=(n >= l.epochsize)

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end
