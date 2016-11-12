for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""
This example learns to classify hand-written digits from the MNIST
dataset (http://yann.lecun.com/exdb/mnist).  There are 60000 training
and 10000 test examples. Each input x consists of 784 pixels
representing a 28x28 image.  The pixel values are normalized to
[0,1]. Each output y is converted to a ten-dimensional one-hot vector
(a vector that has a single non-zero component) indicating the correct
class (0-9) for a given image.  10 is used to represent 0.

You can run the demo using `julia mnist.jl`.  Use `julia mnist.jl
--help` for a list of options.  The dataset will be automatically
downloaded.  By default a softmax model will be trained for 10 epochs.
You can also train a multi-layer perceptron by specifying one or more
--hidden sizes.  The accuracy for the training and test sets will be
printed at every epoch and optimized parameters will be returned.

"""
module MNIST
using Knet,ArgParse,Compat,GZip

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="mnist.jl (c) Deniz Yuret, 2016. Multi-layer perceptron model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=100; help="minibatch size")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.5; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        # These are to experiment with sparse arrays
        # ("--xtype"; help="input array type: defaults to atype")
        # ("--ytype"; help="output array type: defaults to atype")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    w = weights(o[:hidden]...; atype=atype, winit=o[:winit])
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; atype=atype)
    dtst = minibatch(xtst, ytst, o[:batchsize]; atype=atype)
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    if o[:fast]
        @time (train(w, dtrn; lr=o[:lr], epochs=o[:epochs]); gpu()>=0 && Knet.cudaDeviceSynchronize())
    else
        report(0)
        @time for epoch=1:o[:epochs]
            train(w, dtrn; lr=o[:lr], epochs=1)
            report(epoch)
            if o[:gcheck] > 0
                gradcheck(loss, w, first(dtrn)...; gcheck=o[:gcheck])
            end
        end
    end
    return w
end

function predict(w,x)
    for i=1:2:length(w)
        x = w[i]*x .+ w[i+1]
        if i<length(w)-1
            x = relu(x) # max(0,x)
        end
    end
    return x
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = logp(ypred,1) # ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function train(w, dtrn; lr=.5, epochs=10)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = lossgradient(w, x, y)
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
        ynorm = ypred .- log(sum(exp(ypred),1))
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

function loaddata()
    info("Loading MNIST...")
    gzload("train-images-idx3-ubyte.gz")[17:end],
    gzload("t10k-images-idx3-ubyte.gz")[17:end],
    gzload("train-labels-idx1-ubyte.gz")[9:end],
    gzload("t10k-labels-idx1-ubyte.gz")[9:end]
end

function gzload(file; path=Knet.dir("data",file), url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end

if !isdefined(:xtrn)
    (xtrn,xtst,ytrn,ytst)=loaddata()
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia mnist.jl --epochs 10
# julia> MNIST.main("--epochs 10")
!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end # module

# SAMPLE RUN 65f57ff+ Wed Sep 14 10:02:30 EEST 2016
#
# mnist2d.jl (c) Deniz Yuret, 2016. Multi-layer perceptron model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# opts=(:seed,-1)(:batchsize,100)(:hidden,Int64[])(:epochs,10)(:lr,0.5)(:atype,"KnetArray{Float32}")(:gcheck,0)(:winit,0.1)(:fast,true)
# (:epoch,0,:trn,0.079066664f0,:tst,0.0842f0)
#   2.168927 seconds (2.95 M allocations: 115.993 MB, 1.84% gc time)
# (:epoch,10,:trn,0.9195333f0,:tst,0.9158f0)
