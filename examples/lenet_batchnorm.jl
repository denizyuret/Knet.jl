"""
This example learns to classify hand-written digits from the
[MNIST](http://yann.lecun.com/exdb/mnist) dataset.  There are 60000
training and 10000 test examples. Each input x consists of 784 pixels
representing a 28x28 image.  The pixel values are normalized to
[0,1]. Each output y is converted to a ten-dimensional one-hot vector
(a vector that has a single non-zero component) indicating the correct
class (0-9) for a given image.  10 is used to represent 0.

You can run the demo using `julia lenet.jl` at the command line or
`julia> LeNet.main()` at the Julia prompt.  Use `julia lenet.jl
--help` or `julia> LeNet.main("--help")` for a list of options.  The
dataset will be automatically downloaded.  By default the
[LeNet](http://yann.lecun.com/exdb/lenet) convolutional neural network
model will be trained for 10 epochs.  The accuracy for the training
and test sets will be printed at every epoch and optimized parameters
will be returned.
"""
module LeNet

using Knet
include("mnist.jl")
# include("../src/modules.jl")
import .MNIST: minibatch

function predict(w, bmom, x; mode=:train)
    n = 6
    for i=1:3:n
        x = conv4(w[i], x; padding=0)
        x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1], mode=mode)
        x = pool(relu(x))
    end
    x = mat(x)
    for i=n+1:3:length(w)-2
        x = batchnorm(w[i+1:i+2], w[i]*x, bmom[i÷3+1], mode=mode)
        x = relu(x)
    end
    return w[end-1]*x .+ w[end]
end

function loss(w, bmom, x, y; mode=:train)
    ŷ = predict(w, bmom, x, mode=mode)
    ŷ = logp(ŷ, 1)  # ypred .- log(sum(exp(ypred),1))
    return -sum(y .* ŷ) / size(y, 2)
end

function accuracy(w, bmom, data)
    ncorrect = ninstance = nloss = 0
    for (x, y) in data
        ŷ = predict(w, bmom, x, mode=:test)
        ŷ = logp(ŷ, 1)  # ypred .- log(sum(exp(ypred),1))

        nloss += -sum(y .* ŷ)
        ncorrect += sum(y .* (ŷ .== maximum(ŷ,1)))
        ninstance += size(y, 2)
    end
    return (ncorrect/ninstance, nloss/ninstance)
end

function train(w, bmom, data, opt; epochs=1)
    for epoch=1:epochs
        for (x, y) in data
            dw = grad(loss)(w, bmom, x, y; mode=:train)
            for i in 1:length(w)
                Knet.update!(w[i], dw[i], opt[i])
            end
        end
    end
    return w
end

function build_lenet(; atype=KnetArray{Float32}, batchmem=10)
    w = []
    push!(w, xavier(5,5,1,20))
    push!(w, ones(1,1,20,1))
    push!(w, zeros(1,1,20,1))

    push!(w, xavier(5,5,20,50))
    push!(w, ones(1,1,50,1))
    push!(w, zeros(1,1,50,1))

    push!(w, xavier(500,800))
    push!(w, ones(500,1))
    push!(w, zeros(500,1))

    push!(w, xavier(10,500))
    push!(w, zeros(10,1))

    # bmom = [BatchMoments(; momentum=0.9) for _=1:3]
    bmom = [BatchMoments(batchmem) for _=1:3]

    return map(a->convert(atype,a), w), bmom
end

function minibatch4(x, y, batchsize; atype=KnetArray{Float32})
    data = minibatch(x, y, batchsize; atype=atype)
    for i=1:length(data)
        (x,y) = data[i]
        data[i] = (reshape(x, (28,28,1,batchsize)), y)
    end
    return data
end

"""
**Usage Example**:

    julia> include("lenet_batchnorm.jl"); using Knet

    julia> LeNet.main(batchsize=100, optimizer=Momentum(lr=0.1), infotime=5, epochs=30)
"""
function main(;
        seed = -1,
        batchsize = 100,
        optimizer = Momentum(lr=0.01),
        epochs = 100,
        infotime = 1,  # report every `infotime` epochs
        atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32},
        ntrn = 60000, #use only the first `ntrn` samples in training set
        ntst = 10000  #use only the first `ntst` samples in test set
    )

    info("using ", atype)
    seed > 0 && srand(seed)

    isdefined(MNIST,:xtrn) || MNIST.loaddata()
    dtrn = minibatch4(MNIST.xtrn[1:ntrn*784], MNIST.ytrn[1:ntrn], batchsize, atype=atype)
    dtst = minibatch4(MNIST.xtst[1:ntst*784], MNIST.ytst[1:ntst], batchsize, atype=atype)

    w, bmom = build_lenet(atype=atype, batchmem=length(dtrn))
    opt = [deepcopy(optimizer) for _=1:length(w)]

    report(epoch) = println((:epoch, epoch,
                             :trn, accuracy(w, bmom, dtrn),
                             :tst, accuracy(w, bmom, dtst)))

    report(0); tic()
    @time for epoch=1:epochs
        train(w, bmom, dtrn, opt)
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
    end; toq()

    return w
end

end # module
