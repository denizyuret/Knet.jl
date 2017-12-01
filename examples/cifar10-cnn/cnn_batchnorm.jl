#=
This example classifies CIFAR 10(https://www.cs.toronto.edu/~kriz/cifar.html)
dataset using a convolutional neural network with batch normalization

For details of applying batch normalization, see:
"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
Sergey Ioffe  & Christian Szegedy 
https://arxiv.org/abs/1502.03167

The trained architecture has the form of:
conv -> bn -> relu -> pool -> 
conv -> bn -> relu -> pool ->
conv -> bn -> relu ->
fc -> bn  -> relu -> output

=#

for p in ("Knet", )
    (Pkg.installed(p) == nothing) && Pkg.add(p)
end

using Knet
include(Pkg.dir("Knet", "data", "cifar.jl"))

function loaddata()
    info("Loading CIFAR 10...")
    xtrn, ytrn, xtst, ytst, = cifar10()
    #= Subtract mean of each feature
    where each channel is considered as
    a single feature following the CNN
    convention=#
    mn = mean(xtrn, (1,2,4))
    xtrn = xtrn .- mn
    xtst = xtst .- mn
    info("Loaded CIFAR 10")
    return (xtrn, ytrn), (xtst, ytst)
end

# The global device setting (to reduce gpu() calls)
let at = nothing
    global atype
    atype() = (at == nothing) ? (at = (gpu() >= 0 ? KnetArray: Array)) : at
end


##Model definition

#=
Initialization is from
He et al., 2015, 
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
https://arxiv.org/abs/1502.01852
=#
kaiming(et, h, w, i, o) = et(sqrt(2 / (w * h * o))) .* randn(et, h, w, i, o)


function init_model(;et=Float32)
    # Use bnparams() to initialize gammas and betas
    w = Any[
        kaiming(et, 3, 3, 3, 16),    bnparams(et, 16),
        kaiming(et, 3, 3, 16, 32),   bnparams(et, 32),
        kaiming(et, 3, 3, 32, 64),   bnparams(et, 64),
        xavier(et, 100, 8 * 8 * 64), bnparams(et, 100),
        xavier(et, 10, 100),         zeros(et, 10, 1)
    ]
    # Initialize a moments object for each batchnorm
    m = Any[bnmoments() for i = 1:4]
    w = map(atype(), w)
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
    return w[9] * o .+ w[10]
end

function loss(w, m, x, classes)
    ypred = predict(w,m, x)
    return nll(ypred, classes)
end

lossgrad = grad(loss)

# Training
function epoch!(w, m, o, xtrn, ytrn;  mbatch=64)
    data = minibatch(xtrn, ytrn, mbatch;
                   shuffle=true,
                   xtype=atype())
    for (x, y) in data
        g = lossgrad(w, m, x, y)
        update!(w, g, o)
    end
end

# Accuracy computation
function acc(w, m, xtst, ytst; mbatch=64)
    data = minibatch(xtst, ytst, mbatch;
                     partial=true,
                     xtype=atype())
    model = (w, m)
    return accuracy(model, data,
                    (model, x)->predict(model[1], model[2], x);
                    average=true)
end

# TODO: add command line options
function train(;optim=Momentum, epochs=5,
               lr=0.1, oparams...)
    w, m = init_model()
    o = map(_->Momentum(;lr=lr, oparams...), w)
    (xtrn, ytrn), (xtst, ytst) = loaddata()   
    for epoch = 1:epochs
        println("epoch: ", epoch)
        epoch!(w, m, o, xtrn, ytrn)
        println("train accuracy: ", acc(w, m, xtrn, ytrn))
        println("test accuracy: ", acc(w, m, xtst, ytst))
    end
end

endswith(string(PROGRAM_FILE), "cnn_batchnorm.jl") && train()
