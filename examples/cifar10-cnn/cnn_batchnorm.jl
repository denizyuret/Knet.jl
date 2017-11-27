using Knet
include(Pkg.dir("Knet", "data", "cifar.jl"))

function loaddata()
    info("Loading CIFAR 10...")
    xtrn, ytrn, xtst, ytst, = cifar10()
    mn = mean(xtrn, (1,2,4))
    xtrn = xtrn .- mn
    xtst = xtst .- mn
    ytrn = onehot(Int.(ytrn))
    ytst = onehot(Int.(ytst))
    info("Loaded CIFAR 10")
    return (xtrn, ytrn), (xtst, ytst)
end

# TODO: make nll work and remove this encoding
function onehot(ytrnraw, numclass=10; dtype=Float32)
    yonehot = zeros(dtype, numclass, length(ytrnraw))
    for (i, y) in enumerate(ytrnraw)
        yonehot[y, i] = 1.0
    end
    return yonehot
end

# Model definition
kaiming(et, h, w, i, o) = et(sqrt(2 / (w * h * o))) .* randn(et, h, w, i, o)

function init_model(;et=Float32)
    w = Any[
        kaiming(et, 3, 3, 3, 16), bnparam(et, 16),
        kaiming(et, 3, 3, 16, 32), bnparam(et, 32),
        kaiming(et, 3, 3, 32, 64), bnparam(et, 64),
        xavier(et, 100, 8 * 8 * 64), bnparam(et, 100),
        xavier(et, 10, 100), zeros(et, 10, 1)
    ]
    # Initialize a BNMoments object for each batchnorm
    m = Any[BNMoments() for i = 1:4]
    if gpu() >= 0; w = map(KnetArray{et}, w); end
    return w, m
end

function conv_layer(w, m, x; maxpool=true)
    o = conv4(w[1], x; padding=1)
    o = batchnorm(o, m, w[2])
    o = relu.(o)
    if maxpool; o=pool(o); end
    return o
end

function lin_layer(w, m, x)
    o = w[1] * x
    o = batchnorm(o, m, w[2])
    return relu.(o)
end

function predict(w, m, x)
    o = conv_layer(w[1:2] , m[1], x)
    o = conv_layer(w[3:4] , m[2], o)
    o = conv_layer(w[5:6] , m[3], o; maxpool=false)
    o = lin_layer( w[7:8] , m[4], mat(o))
    o = dropout(o, 0.2) # use mild dopouts only
    return w[9] * o .+ w[10]
end

function loss(w, m, x, ygold)
    ypred = predict(w,m, x)
    ynorm = logp(ypred,1)
    return -sum(ygold .* ynorm) / size(ygold,2)
end

lossgrad = grad(loss)

# Training
function epoch!(w, m, p, xtrn, ytrn;  mbatch=64, wdecay=1e-5)
    N = size(ytrn)[end]
    # Shuffling data is recommended when using batch norm
    indices = shuffle(1:N)
    DT = gpu() >= 0 ? KnetArray : Array
    for i = 1:mbatch:(N - N % mbatch) #ignore remainder for consistent batch sizes
        ind = indices[i:i+mbatch-1]
        x = DT(xtrn[:, :, :, ind])
        y = DT(ytrn[:, ind])
        g = lossgrad(w, m, x, y)
        if wdecay>0; g = g .+ wdecay .* w; end
        for (wi, gi, pi) in zip(w, g, p)
            update!(wi, gi, pi) 
        end
    end
end

function acc(w, m, xtst, ytst; mbatch=64)
    ncorrect=0
    ninstance=0.0
    DT = gpu() >= 0 ? KnetArray : Array
    N = size(ytst)[end]
    for i = 1:mbatch:N
        rng = i:min(i+mbatch-1, N) #include remaining for correct acc.
        x, y = DT(xtst[:, :, :, rng]), DT(ytst[:, rng])
        yp = predict(w, m, x)
        ncorrect += sum(y .* (yp .== maximum(yp,1)))
        ninstance += size(y, 2)
    end
    return ncorrect / ninstance
end

function train(;epochs=10)
    w, m = init_model()
    o = map(_->Adam(), w)
    (xtrn, ytrn), (xtst, ytst) = loaddata()   
    for epoch = 1:epochs
        println("epoch: ", epoch)
        epoch!(w, m, o, xtrn, ytrn)
        println("train accuracy: ", acc(w, m, xtrn, ytrn))
        println("test accuracy: ", acc(w, m, xtst, ytst))
    end
end

# TODO: add command line options
function main(ARGS=nothing)
    train()
end

main()
