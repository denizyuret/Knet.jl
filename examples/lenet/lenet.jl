for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","mnist.jl"))


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
using Knet,ArgParse

function predict(w,x)
    n=length(w)-4
    for i=1:2:n
        x = pool(relu.(conv4(w[i],x;padding=0) .+ w[i+1]))
    end
    x = mat(x)
    for i=n+1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end

loss(w,x,ygold) = nll(predict(w,x), ygold)

lossgradient = grad(loss)

function train(w, data; lr=.1, epochs=3, iters=1800)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            update!(w, g, lr=lr)
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
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    println(s.description)
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    if atype <: Array; warn("CPU conv4 support is experimental and very slow."); end

    xtrn,ytrn,xtst,ytst = Main.mnist()
    global dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=atype)
    global dtst = minibatch(xtst, ytst, o[:batchsize]; xtype=atype)
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
PROGRAM_FILE == "lenet.jl" && main(ARGS)

end # module

# SAMPLE RUN 65f57ff+ Wed Sep 14 10:02:30 EEST 2016
#
# lenet.jl (c) Deniz Yuret, 2016. The LeNet model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# opts=(:seed,-1)(:batchsize,100)(:epochs,3)(:lr,0.1)(:gcheck,0)(:fast,true)
# ..................  
# 9.319163 seconds (5.84 M allocations: 277.927 MB, 7.37% gc time)
