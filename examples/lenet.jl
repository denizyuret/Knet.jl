for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet
!isdefined(Main,:MNIST) && include(Knet.dir("examples","mnist.jl"))

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
using Knet,ArgParse,Main
using MNIST: minibatch, accuracy
using Knet: relu_dot

function predict(w,x,n=length(w)-4)
    for i=1:2:n
        x = pool(relu_dot(conv4(w[i],x;padding=0) .+ w[i+1]))
    end
    x = mat(x)
    for i=n+1:2:length(w)-2
        x = relu_dot(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = logp(ypred,1)  # ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function train(w, data; lr=.1, epochs=3, iters=1800)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            for i in 1:length(w)
                # w[i] -= lr * g[i]
                axpy!(-lr, g[i], w[i])
            end
            if (iters -= 1) <= 0
                return w
            end
        end
    end
    return w
end

function weights(;atype=KnetArray{Float32})
    w = Array{Any}(8)
    w[1] = xavier(5,5,1,20)
    w[2] = zeros(1,1,20,1)
    w[3] = xavier(5,5,20,50)
    w[4] = zeros(1,1,50,1)
    w[5] = xavier(500,800)
    w[6] = zeros(500,1)
    w[7] = xavier(10,500)
    w[8] = zeros(10,1)
    return map(a->convert(atype,a), w)
end

function minibatch4(x, y, batchsize; atype=KnetArray{Float32})
    data = minibatch(x,y,batchsize; atype=atype)
    for i=1:length(data)
        (x,y) = data[i]
        data[i] = (reshape(x, (28,28,1,batchsize)), y)
    end
    return data
end

function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="lenet.jl (c) Deniz Yuret, 2016. The LeNet model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=128; help="minibatch size")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--epochs"; arg_type=Int; default=3; help="number of epochs for training")
        ("--iters"; arg_type=Int; default=typemax(Int); help="maximum number of updates for training")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array and float type to use")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    if atype <: Array; warn("CPU conv4 support is experimental and very slow."); end

    isdefined(MNIST,:xtrn) || MNIST.loaddata()
    dtrn = minibatch4(MNIST.xtrn, MNIST.ytrn, o[:batchsize]; atype=atype)
    dtst = minibatch4(MNIST.xtst, MNIST.ytst, o[:batchsize]; atype=atype)
    w = weights(atype=atype)
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))

    if o[:fast]
        @time (train(w, dtrn; lr=o[:lr], epochs=o[:epochs], iters=o[:iters]); gpu()>=0 && Knet.cudaDeviceSynchronize())
    else
        report(0)
        iters = o[:iters]
        for epoch=1:o[:epochs]
            @time train(w, dtrn; lr=o[:lr], epochs=1, iters=iters)
            report(epoch)
            if o[:gcheck] > 0
                gradcheck(loss, w, first(dtrn)...; gcheck=o[:gcheck], verbose=true)
            end
            if (iters -= length(dtrn)) <= 0; break; end
        end
    end
    return w
end


# This allows both non-interactive (shell command) and interactive calls like:
# $ julia lenet.jl --epochs 10
# julia> LeNet.main("--epochs 10")
if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "lenet.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end # module

# SAMPLE RUN 65f57ff+ Wed Sep 14 10:02:30 EEST 2016
#
# lenet.jl (c) Deniz Yuret, 2016. The LeNet model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# opts=(:seed,-1)(:batchsize,100)(:epochs,3)(:lr,0.1)(:gcheck,0)(:fast,true)
# ..................  
# 9.319163 seconds (5.84 M allocations: 277.927 MB, 7.37% gc time)
