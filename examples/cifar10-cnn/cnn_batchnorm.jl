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
        kaiming(et, 3, 3, 3, 16), ones(et, 1, 1, 16, 1), zeros(et, 1, 1, 16, 1),
        kaiming(et, 3, 3, 16, 32), ones(et, 1, 1, 32, 1), zeros(et, 1, 1, 32, 1),
        kaiming(et, 3, 3, 32, 64), ones(et, 1, 1, 64, 1), zeros(et, 1, 1, 64, 1),
        xavier(et, 100, 8 * 8 * 64), ones(et, 100, 1), zeros(et, 100, 1),
        xavier(et, 10, 100), zeros(et, 10, 1)
    ]
    # Initialize a BNMoments object for each batchnorm
    m = Any[BNMoments() for i = 1:4]
    if gpu() >= 0; w = map(KnetArray{et}, w); end
    return w, m
end

function conv_layer(w, m, x; training=true, maxpool=true)
    o = conv4(w[1], x; padding=1)
    o = batchnorm(m, w[2], w[3], o;
                  training=training)
    o = relu.(o)
    if maxpool; o=pool(o); end
    return o
end

function lin_layer(w, m, x; training=true)
    o = w[1] * x
    o = batchnorm(m, w[2], w[3], o; training=training)
    return relu.(o)
end

function predict(w, m, x; training=true)
    o = conv_layer(w[1:3] , m[1], x; training=training)
    o = conv_layer(w[4:6] , m[2], o; training=training)
    o = conv_layer(w[7:9] , m[3], o; training=training, maxpool=false)
    o = lin_layer(w[10:12], m[4], mat(o); training=training)
    w[end-1] * o .+ w[end]
end

function loss(w, m, x, ygold)
    ypred = predict(w,m, x)
    ynorm = logp(ypred,1)
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgrad = grad(loss)

# Training
function epoch!(w, m, p, xtrn, ytrn;  mbatch=64)
    N = size(ytrn)[end]
    indices = shuffle(1:N)
    DT = gpu() >= 0 ? KnetArray : Array
    for i = 1:mbatch:(N - N % mbatch)
        ind = indices[i:i+mbatch-1]
        x = DT(xtrn[:, :, :, ind])
        y = DT(ytrn[:, ind])
        g = lossgrad(w, m, x, y)
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
        rng = i:min(i+mbatch-1, N)
        x, y = DT(xtst[:, :, :, rng]), DT(ytst[:, rng])
        yp = predict(w, m, x; training=false)
        ncorrect += sum(y .* (yp .== maximum(yp,1)))
        ninstance += size(y, 2)
    end
    return ncorrect / ninstance
end

function train(;epochs=5)
    w, m = init_model()
    o = map(_->Momentum(;lr=0.1), w)
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
