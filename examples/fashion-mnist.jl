for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""

This example learns to classify images of fashion products(trousers, shirts, bags...) 
from the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.  
There are 60000 training and 10000 test examples. Each input x 
consists of 784 pixels representing a 28x28 image. The pixel values are 
normalized to [0,1]. Each output y is converted to a ten-dimensional 
one-hot vector (a vector that has a single non-zero component) indicating 
the correct class (0-9) for a given image. 10 is used instead of 0.
Labels and descriptions are shown below.

Label   Description
0/10    T-shirt/top
1       Trouser
2       Pullover
3       Dress
4       Coat
5       Sandal
6       Shirt
7       Sneaker
8       Bag
9       Ankle boot

You can run the demo using `julia fashion-mnist.jl` on the command line or
by first including `julia> include("fashion-mnist.jl")` and typing `julia> FashionMNIST.main()` 
at the Julia prompt.  Options can be used like `julia fashion-mnist.jl --epochs 3` 
or `julia> FashionMNIST.main("--epochs 3")`. Use `julia fashion-mnist.jl --help` 
for a list of options.  The dataset will be automatically downloaded.  
By default a softmax model will be trained for 10 epochs. You can also 
train a multi-layer perceptron by specifying one or more --hidden sizes. 
The accuracy for the training and test sets will be printed at every epoch 
and optimized parameters will be returned.

"""
module FashionMNIST
using Knet,ArgParse,Compat,GZip
using Knet: relu_dot

function predict(w,x; pdrop=0)
    for i=1:2:length(w)
        x = w[i]*dropout(x, pdrop) .+ w[i+1]
        if i<length(w)-1
            x = relu_dot(x) # max(0,x)
        end
    end
    return x
end

function loss(w,x,ygold; pdrop=0)
    ypred = predict(w,x; pdrop=pdrop)
    ynorm = logp(ypred,1) # ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function train(w, dtrn; lr=.5, epochs=10, pdrop=0)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = lossgradient(w, x, y; pdrop=pdrop)
            for i in 1:length(w)
                # w[i] -= lr * g[i]
                axpy!(-lr, g[i], w[i])
            end
        end
    end
    return w
end

function accuracy(w, dtst, pred=predict)
    ncorrect = ninstance = nloss = 0
    for (x, ygold) in dtst
        ypred = pred(w, x)
        ynorm = logp(ypred,1) # ypred .- log(sum(exp(ypred),1))
        nloss += -sum(ygold .* ynorm)
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold,2)
    end
    return (ncorrect/ninstance, nloss/ninstance)
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

function loaddata()
    global xtrn,ytrn,xtst,ytst
    info("Loading Fashion-MNIST...")
    xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
    xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
    ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
    ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
    info("Loaded Fashion-MNIST...")
end

function gzload(file; path=Knet.dir("data/fashion-mnist",file), url="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/$file")
    download_dir = Knet.dir("data/fashion-mnist")
    ispath(download_dir) || mkpath(download_dir)
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end

function minibatch(x, y, batchsize; atype=Array{Float32}, xrows=784, yrows=10, xscale=255)
    xbatch(a)=convert(atype, reshape(a./xscale, xrows, div(length(a),xrows)))
    ybatch(a)=(a[a.==0]=10; convert(atype, sparse(convert(Vector{Int},a),1:length(a),one(eltype(a)),yrows,length(a))))
    xcols = div(length(x),xrows)
    xcols == length(y) || throw(DimensionMismatch())
    data = Any[]
    for i=1:batchsize:xcols-batchsize+1
        j=i+batchsize-1
        push!(data, (xbatch(x[1+(i-1)*xrows:j*xrows]), ybatch(y[i:j])))
    end
    return data
end

function main(args="")
    s = ArgParseSettings()
    s.description="fashion-mnist.jl (c) 2017 Adapted by Emre Unal based on Deniz Yuret’s MNIST example(https://github.com/denizyuret/Knet.jl/tree/master/examples/mnist.jl).\nMulti-layer perceptron model on the Fashion-MNIST dataset from https://github.com/zalandoresearch/fashion-mnist.\n"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=100; help="minibatch size")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.15; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        ("--dropout"; arg_type=Float64; default=0.5; help="Dropout probability.")
        # These are to experiment with sparse arrays
        # ("--xtype"; help="input array type: defaults to atype")
        # ("--ytype"; help="output array type: defaults to atype")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    if !o[:fast]
        println(s.description)
        println("opts=",[(k,v) for (k,v) in o]...)
    end
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    w = weights(o[:hidden]...; atype=atype, winit=o[:winit])
    if !isdefined(FashionMNIST,:xtrn); loaddata(); end
    global dtrn = minibatch(xtrn, ytrn, o[:batchsize]; atype=atype)
    global dtst = minibatch(xtst, ytst, o[:batchsize]; atype=atype)
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    if o[:fast]
        (train(w, dtrn; lr=o[:lr], epochs=o[:epochs], pdrop=o[:dropout]); gpu()>=0 && Knet.cudaDeviceSynchronize())
    else
        report(0)
        @time for epoch=1:o[:epochs]
            train(w, dtrn; lr=o[:lr], epochs=1)
            report(epoch)
            if o[:gcheck] > 0
                gradcheck(loss, w, first(dtrn)...; gcheck=o[:gcheck], verbose=true)
            end
        end
    end
    return w
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia mnist.jl --epochs 10
# julia> FashionMNIST.main("--epochs 10")
if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "fashion-mnist.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end # module
