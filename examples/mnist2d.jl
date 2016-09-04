isdefined(:MNIST) || include("mnist.jl")

module MNIST2D
using Knet,AutoGrad,ArgParse
using Main.MNIST: xtrn,ytrn,xtst,ytst
using Base.LinAlg: axpy!

function main(args=ARGS)
    global w, dtrn, dtst
    s = ArgParseSettings()
    s.description="mnist2d.jl (c) Deniz Yuret, 2016. Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1)
        ("--batchsize"; arg_type=Int; default=100)
        ("--epochs"; arg_type=Int; default=20)
        ("--hidden"; nargs='+'; arg_type=Int)
        ("--lr"; arg_type=Float64; default=0.5)
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)
    o[:seed] > 0 && srand(o[:seed])
    w = weights(o[:hidden]...)
    dtrn = minibatch(xtrn, ytrn, o[:batchsize])
    dtst = minibatch(xtst, ytst, o[:batchsize])
    println((:epoch,0,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    @time train(w, dtrn; lr=o[:lr], epochs=o[:epochs])
    println((:epoch,o[:epochs],:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
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

function minibatch(x, y, batchsize)
    x = reshape(x, (div(length(x),size(x,ndims(x))), size(x,ndims(x))))
    data = Any[]
    for i=1:batchsize:size(x,2)-batchsize+1
        j=i+batchsize-1
        push!(data, (KnetArray(x[:,i:j]), KnetArray(y[:,i:j])))
    end
    return data
end

function weights(h...)
    w = Any[]
    x = 28*28
    for y in [h..., 10]
        push!(w, KnetArray(convert(Array{Float32}, 0.1*randn(y,x))))
        push!(w, KnetArray(zeros(Float32, y, 1)))
        x = y
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



# This allows both non-interactive (shell command) and interactive calls like:
# julia> mnist2d("--epochs 10")
!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end # module

