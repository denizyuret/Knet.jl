"""
This example learns to classify hand-written digits from the MNIST
dataset.  There are 60000 training and 10000 test examples. Each input
x consists of 784 pixels representing a 28x28 image.  The pixel values
have been normalized to [0,1]. Each output y is a ten-dimensional
one-hot vector (a vector that has a single non-zero component)
indicating the correct class (0-9) for a given image.

To run the demo, simply `include("mnist.jl")` and run `MNIST.train()`.
The dataset will be automatically downloaded.  You can provide the
initial weights as an optional argument to `train`, which should have
the form [w0,b0,w1,b1,...] where wi (with size = output x input) is
the weight matrix and bi (with size = output) is the bias vector for
layer i.  The function `MNIST.weights(h...)` can be used to create
random starting weights for a neural network with hidden sizes (h...).
If not specified, default weights are created using `MNIST.weights()`
which correspond to a 0 hidden layer network, i.e. a softmax model.
`train` also accepts the following keyword arguments: `lr` specifies
the learning rate, `epochs` gives number of epochs.  The cross entropy
loss and accuracy for the train and test sets will be printed at every
epoch and optimized parameters will be returned.
"""
module MNIST
using Knet,ArgParse,Compat,GZip

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
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        # These are to experiment with sparse arrays
        # ("--xtype"; help="input array type: defaults to atype")
        # ("--ytype"; help="output array type: defaults to atype")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    w = weights(o[:hidden]...; atype=atype, winit=o[:winit])
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=atype, ytype=atype)
    dtst = minibatch(xtst, ytst, o[:batchsize]; xtype=atype, ytype=atype)
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

function accuracy(w, dtst)
    ncorrect = ninstance = 0
    for (x, ygold) in dtst
        ypred = predict(w, x)
        ncorrect += sum((ypred .== maximum(ypred,1)) .* (ygold .== maximum(ygold,1)))
        ninstance += size(ygold,2)
    end
    return ncorrect/ninstance
end

function weights(h...; atype=Array{Float32}, winit=0.1)
    w = Any[]
    x = 28*28
    for y in [h..., 10]
        push!(w, convert(atype, winit*randn(y,x)))
        push!(w, convert(atype, zeros(y, 1)))
        x = y
    end
    return w
end

function minibatch(x, y, batchsize; xtype=Array{Float32}, ytype=Array{Float32}, xrows=784, yrows=10, xscale=255.)
    xbatch(a)=convert(xtype, reshape(a,xrows,div(length(a),xrows)))./xscale
    ybatch(a)=(a[a.==0]=10; convert(ytype, sparse(convert(Vector{Int},a),1:length(a),one(eltype(a)),yrows,length(a))))
    xcols = div(length(x),xrows)
    xcols == length(y) || throw(DimensionMismatch())
    data = Any[]
    for i=1:batchsize:xcols-batchsize+1
        j=i+batchsize-1
        push!(data, (xbatch(x[1+(i-1)*xrows:j*xrows]), ybatch(y[i:j])))
    end
    return data
end

function loaddata()
    info("Loading MNIST...")
    gzread("train-images-idx3-ubyte.gz")[17:end],
    gzread("t10k-images-idx3-ubyte.gz")[17:end],
    gzread("train-labels-idx1-ubyte.gz")[9:end],
    gzread("t10k-labels-idx1-ubyte.gz")[9:end]
end

function gzread(file; dir=Pkg.dir("Knet/data/"), url="http://yann.lecun.com/exdb/mnist/")
    path = dir*file
    isfile(path) || download(url*file, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end

if !isdefined(:xtrn)
    (xtrn,xtst,ytrn,ytst)=loaddata()
end

# This allows both non-interactive (shell command) and interactive calls like:
# julia> mnist2d("--epochs 10")
!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end # module

