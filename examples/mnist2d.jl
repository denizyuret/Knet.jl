isdefined(:MNIST) || include("mnist.jl")

module MNIST2D
using Knet,AutoGrad,ArgParse
using Main.MNIST: xtrn,ytrn,xtst,ytst

function main(args=ARGS)
    global w, dtrn, dtst
    s = ArgParseSettings()
    s.description="mnist2d.jl (c) Deniz Yuret, 2016. Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=100; help="minibatch size")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--hidden"; nargs='+'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.5; help="learning rate")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    global w = weights(o[:hidden]...; atype=atype)
    global dtrn = minibatch(xtrn, ytrn, o[:batchsize]; atype=atype)
    global dtst = minibatch(xtst, ytst, o[:batchsize]; atype=atype)
    println((:epoch,0,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    if o[:fast]
        @time train(w, dtrn; lr=o[:lr], epochs=o[:epochs])
        println((:epoch,o[:epochs],:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    else
        @time for epoch=1:o[:epochs]
            train(w, dtrn; lr=o[:lr], epochs=1)
            println((:epoch,epoch,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
        end
    end
    return w
end

function predict(w,x)
    for i=1:2:length(w)
        x = w[i]*x .+ w[i+1]
        if i<length(w)-1
            x = max(0,x)
        end
    end
    return x
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function train(w, dtrn; lr=.1, epochs=20)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = lossgradient(w, x, y)
            for i in 1:length(w)
                w[i] -= lr * g[i]
            end
        end
    end
    return w
end

function minibatch(x, y, batchsize; atype=Array{Float32})
    x = reshape(x, (div(length(x),size(x,ndims(x))), size(x,ndims(x))))
    data = Any[]
    for i=1:batchsize:size(x,2)-batchsize+1
        j=i+batchsize-1
        push!(data, (convert(atype,x[:,i:j]), convert(atype,y[:,i:j])))
    end
    return data
end

function weights(h...; atype=Array{Float32})
    w = Any[]
    x = 28*28
    for y in [h..., 10]
        push!(w, convert(atype, 0.1*randn(y,x)))
        push!(w, convert(atype, zeros(y, 1)))
        x = y
    end
    return w
end

function accuracy(w, dtst)
    ncorrect = ninstance = 0
    for (x, ygold) in dtst
        ypred = predict(w, x)
        ncorrect += sum((ypred .== maximum(ypred,1)) .* (ygold .== maximum(ygold,1)))
        ninstance += size(ygold,2)
    end
    return ncorrect/ninstance
end

# This allows both non-interactive (shell command) and interactive calls like:
# julia> mnist2d("--epochs 10")
!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end # module

