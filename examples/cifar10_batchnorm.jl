# A temporary example that will be edited
include("../src/batchnorm.jl")

using MLDatasets

# Returns a tuple of two tuples: training and test data and labels
# Data
function cifar10(dir=nothing, onehot=true; dtype = Float32)
    dir = (dir == nothing) ? string(pwd(),"/cifar10") : dir
    loader = MLDatasets.CIFAR10
    (xtr, ytr) = loader.traindata(dir)
    (xts, yts) = loader.testdata(dir)
    xtr = convert(Array{dtype}, xtr)
    xts = convert(Array{dtype}, xts)
    if onehot
        ytr = toonehot(ytr+1, 10)
        yts = toonehot(yts+1, 10)
    end
    return ((xtr, ytr), (xts, yts))
end

function toonehot(ytrnraw, numclass; dtype=Float32)
    println("One hot encoding...")
    yonehot = zeros(dtype, numclass, length(ytrnraw))
    # println(ytrnraw)
    for (i, y) in enumerate(ytrnraw)
      # println(i," ", y)
        yonehot[y, i] = 1.0
    end
    println("One hot encoding done.")
    #y[ytrnraw[:], 1:length(ytrnraw)] = 1.0
    return yonehot
end

function loaddata()
    println("Loading data...")
    dtr, dts = cifar10()
    println("Data is read...")
    (xtrn, ytrn) = dtr
    (xtst, ytst) = dts
    mnt = mean(xtrn, (1, 2, 4))
    xtrn .-= mnt
    xtst .-= mnt
    return (xtrn, ytrn), (xtst, ytst)
end

function next_batch(x, y, bs)
    batch_indices = rand(1:size(x, 4), bs)
    x_, y_ =  x[:, :, :, batch_indices], y[:, batch_indices]
    if gpu() >= 0
        ka = KnetArray{eltype(x)}
        x_, y_ = ka(x_), ka(y_)
    end
    return x_, y_
end

function cinit(h,w,i,o; etype=Float32)
    w = sqrt(2 / (h * w * o)) .* randn(etype, h, w, i, o)
    if gpu() >= 0
        w = KnetArray{etype}(w)
    end
    return w
end

function bninit(input, ndims; etype=Float32)
    # parameters
    shape = (ndims == 4) ? (1,1,input,1) : (input,1)
    gamma = ones(etype, shape...)
    beta = zeros(etype, shape...)
    # moments
    moments = BNMoments()
    cache = BNCache()
    if gpu() >= 0
        gamma = KnetArray{etype}(gamma)
        beta = KnetArray{etype}(beta)
    end
    return (gamma, beta), moments, cache
end

bn4init(input; o...) = bninit(input, 4; o...)
bn2init(input; o...) = bninit(input, 2; o...)

function fcinit(output, input; etype=Float32, bias=false)
    w = xavier(etype, output, input)
    w = (gpu() >= 0) ? KnetArray{etype}(w) : w
    if bias
        b = zeros(etype, output, 1)
        b = (gpu() >= 0) ? KnetArray{etype}(b) : b
        return w, b
    else
        return w
    end
end

# Model
function init_model()
    ws = []
    ms = []
    cs = []
    ich = 3
    # 3 conv layers
    for ch in [16, 32, 64]
        # add convolution
        push!(ws, cinit(5,5, ich, ch))
        ich = ch
        # add batchnorm
        w, m, c = bn4init(ch)
        push!(ws, w...)
        push!(ms, m)
        push!(cs, c)
    end
    # 1 fc hidden layer
    push!(ws, fcinit(100, 64ich))
    w,m,c = bn2init(100)
    push!(ws, w...)
    push!(ms, m)
    push!(cs, c)
    # Output layer
    push!(ws, fcinit(10, 100; bias=true)...)
    return ws, ms, cs
end

function cbrp(w, x, m, c; training=true, maxpool=true)
    o = conv4(w[1], x; padding=2)
    o = batchnorm4(w[2], w[3], o;
                   moments=m,
                   cache=c,
                   training=training)
    return maxpool ? pool(relu.(o)) : relu.(o)
    
end

function fbr(w, x, m, c; training=true)
    if ndims(x) > 2; x=mat(x); end
    o = w[1] * x
    o = batchnorm2(w[2], w[3], o;
                   moments=m,
                   cache=c,
                   training=training)
    return relu.(o)
end

function predict(w, x, m, c; training=true)
    x = cbrp(w[1:3] , x, m[1], c[1]; training=training)
    x = cbrp(w[4:6] , x, m[2], c[2]; training=training)
    x = cbrp(w[7:9] , x, m[3], c[3]; training=training, maxpool=false)
    x = fbr(w[10:12], x, m[4], c[4]; training=training)
    return w[end-1] * x .+ w[end]
end

function loss(w, x, m, c, ygold; o...)
    y = predict(w, x, m, c)
    return -sum(ygold .* logp(y, 1)) ./ size(ygold, 2)
end

lossgrad = grad(loss)

# Training
function accuracy(w,m,c,dtst)
    dtype = isa(w[1], KnetArray) ? KnetArray : Array
    ncorrect = 0
    ninstance = 0
    nloss = 0
    X, Y = dtst
    bsize = 256
    for i = 1:bsize:size(Y,2)
        ending = min(i+bsize-1, size(X,4))
        x = convert(dtype, X[:, :, :, i:ending])
        ygold = convert(dtype, Y[:, i:ending])
        #println("Accuracy iter ", i)
        ypred = predict(w, x, m, c; training=false)
        #nloss += result_loss(ypred, ygold)
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold, 2)
        #nloss_count += 1
    end
    #println(ncorrect, " ", ninstance," ", nloss, " ", nloss_count)
    return (ncorrect / ninstance)#, nloss / nloss_count)
end

function train(;iters=10000)
    @inline report() = begin
        info("Computing Accuracy")
        println("Training accuracy: ", accuracy(w,m,c,dtrn))
        println("Test accuracy: ", accuracy(w,m,c,dtst))
        println()
    end
    w, m, c = init_model()
    dtrn, dtst = loaddata()
    opt = [Sgd(lr=.1) for _ in w]
    for i = 1:iters
        ((i-1) % 200 == 0) && report()
        ((i-1) % 100 == 0) && println("iter: ", i)
        x, y = next_batch(dtrn[1], dtrn[2], 64)
        grads = lossgrad(w, x, m, c, y; training=true)
        for (p, o, g) in zip(w, opt, grads)
            update!(p, g, o)
        end
    end
    println("Training is over")
    report()
    return w, m
end

train()
