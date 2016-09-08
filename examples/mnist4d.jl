# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# 4-D convolution test

isdefined(:MNIST) || include("mnist.jl")

module MNIST4D
using Knet,AutoGrad,CUDNN,ArgParse

# CUDNN supports CudaArrays, here is a hack until we implement KnetArray support

convert{T}(::Type{CudaPtr}, p::KnetPtr{T})=CudaPtr(p.ptr)
convert{T,N}(::Type{CudaArray}, x::KnetArray{T,N})=CudaArray{T,N}(CudaPtr(x.ptr), size(x), x.dev)

# Define some new primitives: conv4 and pool4

function conv4{T}(x::KnetArray{T},w::KnetArray{T}; o...)
    cx = CudaArray(x)
    cw = CudaArray(w)
    ydims = cudnnGetConvolutionNdForwardOutputDim(cx,cw; o...)
    y = similar(x, ydims)
    cy = CudaArray(y)
    cudnnConvolutionForward(cx, cw, cy; o...)
    return y
end

function conv4x{T}(x::KnetArray{T},w::KnetArray{T},dy::KnetArray{T}; o...)
    dx = similar(x)
    cw = CudaArray(w)
    cdx = CudaArray(dx)
    cdy = CudaArray(dy)
    cudnnConvolutionBackwardData(cw,cdy,cdx; o...)
    return dx
end

function conv4w{T}(x::KnetArray{T},w::KnetArray{T},dy::KnetArray{T}; o...)
    dw = similar(w)
    cx = CudaArray(x)
    cdy = CudaArray(dy)
    cdw = CudaArray(dw)
    cudnnConvolutionBackwardFilter(cx,cdy,cdw; o...)
    return dw
end

@primitive  conv4(x,w; o...),dy  conv4x(x,w,dy;o...)  conv4w(x,w,dy;o...)

function pool4{T}(x::KnetArray{T}; o...)
    pd = PD(ndims=ndims(x), o...)
    cx = CudaArray(x)
    ydims = cudnnGetPoolingNdForwardOutputDim(pd, cx)
    y = similar(x, ydims)
    cy = CudaArray(y)
    cudnnPoolingForward(cx, cy; o...)
    return y
end

function pool4x{T}(x::KnetArray{T},y::KnetArray{T},dy::KnetArray{T}; o...)
    dx = similar(x)
    cx = CudaArray(x)
    cy = CudaArray(y)
    cdy = CudaArray(dy)
    cdx = CudaArray(dx)
    cudnnPoolingBackward(cy,cdy,cx,cdx; o...)
    return dx
end

@primitive  pool4(x;o...),dy,y  pool4x(x,y,dy;o...)

function predict(w,x0)          # LeNet model
    x1 = pool4(max(0, conv4(w[1],x0) .+ w[2]))
    x2 = pool4(max(0, conv4(w[3],x1) .+ w[4]))
    x3 = max(0, w[5]*x2 .+ w[6])
    x4 = w[7]*x3 .+ w[8]
end

function weights(;ftype=Float32,atype=KnetArray,winit=0.1) # TODO: xavier
    w = Array(Any,8)
    w[1] = randn(Float32,5,5,1,20)*winit
    w[2] = zeros(Float32,1,1,20,1)
    w[3] = randn(Float32,5,5,20,50)*winit
    w[4] = zeros(Float32,1,1,50,1)
    w[5] = randn(Float32,500,100)*winit
    w[6] = zeros(Float32,500,1)
    w[7] = randn(Float32,10,500)*winit
    w[8] = zeros(Float32,10,1)
    return map(a->convert(atype,a), w)
end

function minibatch(x, y, batchsize; atype=KnetArray)
    data = Any[]
    for i=1:batchsize:size(x,2)-batchsize+1
        j=i+batchsize-1
        xi = convert(atype, reshape(x[:,i:j],(28,28,1,batchsize)))
        yi = convert(atype, y[:,i:j])
        push!(data, (xi, yi))
    end
    return data
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function accuracy(w, dtst)
    ncorrect = ninstance = 0
    for (x, ygold) in dtst
        ypred = predict(w, x)
        ncorrect += sum((ypred .== maximum(ypred,1)) .* (ygold .== maximum(ygold,1)))
        ninstance += size(ygold,2)
    end
    return ncorrect/ninstance
end

function train(w, data; lr=.1, epochs=20)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            for i in 1:length(w)
                w[i] -= lr * g[i]
            end
        end
    end
    return w
end

Base.randn(T::Type, dims::Dims) = convert(Array{T}, randn(dims))
Base.randn(T::Type, d1::Integer, dims::Integer...) = randn(T, convert(Tuple{Vararg{Int}}, (d1,dims...)))

function main(args=ARGS)
    info("Testing lenet (convolutional net) on MNIST")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--batchsize"; arg_type=Int; default=100)
        ("--lr"; arg_type=Float64; default=0.1)
        ("--epochs"; arg_type=Int; default=3)
        #TODO: ("--gcheck"; arg_type=Int; default=0), --atype, --winit, --fast
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])

    global dtrn = minibatch(MNIST.xtrn, MNIST.ytrn, o[:batchsize])
    global dtst = minibatch(MNIST.xtst, MNIST.ytst, o[:batchsize])
    global w = weights()

    println((:epoch,0,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    if o[:fast]
        @time train(w, dtrn; lr=o[:lr], epochs=o[:epochs])
        println((:epoch,o[:epochs],:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    else
        @time for epoch=1:o[:epochs]
            train(w, dtrn; lr=o[:lr], epochs=1)
            println((:epoch,epoch,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
        end
    end
    return w
end

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end # module


### DEAD CODE:

# function train(f, data, loss; losscnt=nothing, maxnorm=nothing)
#     for (x,ygold) in data
#         ypred = forw(f, x)
#         back(f, ygold, loss)
#         update!(f)
#         losscnt[1] += loss(ypred, ygold); losscnt[2] += 1
#         w=wnorm(f); w > maxnorm[1] && (maxnorm[1]=w)
#         g=gnorm(f); g > maxnorm[2] && (maxnorm[2]=g)
#     end
# end

# function test(f, data, loss)
#     sumloss = numloss = 0
#     for (x,ygold) in data
#         ypred = forw(f, x)
#         sumloss += loss(ypred, ygold)
#         numloss += 1
#     end
#     sumloss / numloss
# end

# function getgrad(f, data, loss)
#     (x,ygold) = first(data)
#     ypred = forw(f, x)
#     back(f, ygold, loss)
#     loss(ypred, ygold)
# end

# function getloss(f, data, loss)
#     (x,ygold) = first(data)
#     ypred = forw(f, x)
#     loss(ypred, ygold)
# end

# single batch for training for quick debug
# dtrn1 = ItemTensor(reshape(MNIST.xtrn,28,28,1,div(length(MNIST.xtrn),28*28)), MNIST.ytrn; batch=nbatch,epoch=nbatch)
# @date @show test(lenet, dtrn1) # to initialize weights
# @date @show train(lenet, dtrn1)
# @test isequal(x0,dtrn.data[1])
# @test isequal(y0,dtrn.data[2])

#using Knet: params, isapprox2

# lenet = Net(Conv(20,5), Bias(), Relu(), Pool(2),
#             Conv(50,5), Bias(), Relu(), Pool(2),
#             Mmul(500), Bias(), Relu(),
#             Mmul(10), Bias(), Soft(), SoftLoss())

#lenet0 = deepcopy(lenet)
#lemlp = deepcopy(lenet.op)
#@show map(isequal, params(lenet), params(lemlp))

#@date train(lemlp, csub(dtrn.data[1],1:100), csub(dtrn.data[2],1:100); batch=nbatch)
#@show map(isequal, params(lenet), params(lemlp))
#@show map(isapprox, params(lenet), params(lemlp))

    # @show (i,1,map(vecnorm,params(lenet)),map(difnorm,params(lenet)))
    # @show (i,1,map(vecnorm,params(lemlp)),map(difnorm,params(lemlp)))
    # @test all(map(isequal, params(lenet), params(lemlp)))
# @show (0,0,map(vecnorm,params(lenet)),map(difnorm,params(lenet)))
# @show (0,0,map(vecnorm,params(lemlp)),map(difnorm,params(lemlp)))


    # @date train(lemlp, dtrn.data[1], dtrn.data[2]; batch=nbatch)
    # these fail
    # @show map(isequal, params(lenet), params(lemlp))
    # @show map(isapprox, params(lenet), params(lemlp))
             # accuracy(dtrn.data[2], predict(lemlp, dtrn.data[1])),
             # accuracy(dtst.data[2], predict(lemlp, dtst.data[1]))))

### SAMPLE RUN:

# [dy_052@hpc3010 examples]$ julia mnist4d.jl
# INFO: Loading MNIST...
#   5.658037 seconds (362.46 k allocations: 503.185 MB, 1.83% gc time)
# INFO: Testing lenet
# 2015-10-02T14:32:31 @show (epoch,lwg...)
# (epoch,lwg...) = (1,1.8373411f0,15.339206f0,13.506349f0)
#   0.098889 seconds (49.55 k allocations: 2.261 MB, 4.69% gc time)
# 2015-10-02T14:32:31 @show accuracy(lenet,dtrn)
# accuracy(lenet,dtrn) = 0.8724166666666666
#   1.705410 seconds (1.54 M allocations: 81.031 MB, 1.04% gc time)
# 2015-10-02T14:32:33 @show accuracy(lenet,dtst)
# accuracy(lenet,dtst) = 0.8802
#   0.212678 seconds (196.73 k allocations: 10.577 MB, 2.62% gc time)
# 2015-10-02T14:32:37 @show (epoch,lwg...)
# (epoch,lwg...) = (2,0.16057172f0,17.989243f0,12.976568f0)
#   0.000041 seconds (50 allocations: 1.766 KB)
# 2015-10-02T14:32:37 @show accuracy(lenet,dtrn)
# accuracy(lenet,dtrn) = 0.9530333333333333
#   1.256893 seconds (1.16 M allocations: 63.102 MB, 1.32% gc time)
# 2015-10-02T14:32:38 @show accuracy(lenet,dtst)
# accuracy(lenet,dtst) = 0.9543
#   0.212085 seconds (196.76 k allocations: 10.578 MB, 2.61% gc time)
# 2015-10-02T14:32:43 @show (epoch,lwg...)
# (epoch,lwg...) = (3,0.08003744f0,19.31503f0,8.413661f0)
#   0.000042 seconds (50 allocations: 1.766 KB)
# 2015-10-02T14:32:43 @show accuracy(lenet,dtrn)
# accuracy(lenet,dtrn) = 0.9698833333333333
#   1.254705 seconds (1.16 M allocations: 63.105 MB, 1.32% gc time)
# 2015-10-02T14:32:44 @show accuracy(lenet,dtst)
# accuracy(lenet,dtst) = 0.9698
#   0.211996 seconds (196.76 k allocations: 10.578 MB, 2.60% gc time)

    # prog = quote
    #     x  = input()
    #     w1 = par(5,5,0,20)
    #     c1 = conv(w1,x)
    #     b1 = par(0)
    #     d1 = add(b1,c1)
    #     r1 = relu(d1)
    #     p1 = pool(r1; window=2)
    #     w2 = par(5,5,0,50)
    #     c2 = conv(w2,p1)
    #     b2 = par(0)
    #     d2 = add(b2,c2)
    #     r2 = relu(d2)
    #     p2 = pool(r2; window=2)
    #     w3 = par(500,0)
    #     a3 = dot(w3,p2)
    #     b3 = par(0)
    #     c3 = add(b3,a3)
    #     d3 = relu(c3)
    #     w4 = par(10,0)
    #     a4 = dot(w4,d3)
    #     b4 = par(0)
    #     c4 = add(b4,a4)
    #     p = soft(c4)
    # end

### Sample run with Gaussian init:
# (epoch,lwg...) = (1,1.8372313f0,15.332554f0,11.595623f0)
# 1 - test(lenet,dtrn; loss=zeroone) = 0.8728666666666667
# 1 - test(lenet,dtst; loss=zeroone) = 0.8805
# (epoch,lwg...) = (2,0.16045235f0,17.99744f0,12.73949f0)
# 1 - test(lenet,dtrn; loss=zeroone) = 0.9522166666666667
# 1 - test(lenet,dtst; loss=zeroone) = 0.9538000000000001
# (epoch,lwg...) = (3,0.08008432f0,19.315722f0,8.510152f0)
# 1 - test(lenet,dtrn; loss=zeroone) = 0.9704833333333335
# 1 - test(lenet,dtst; loss=zeroone) = 0.9701000000000001

### Sample run with default w=Gaussian, b=Constant, c=Xavier init:
# (epoch,lwg...) = (1,0.34929836f0,24.005178f0,15.993354f0)
# 1 - test(lenet,dtrn; loss=zeroone) = 0.9650666666666667
# 1 - test(lenet,dtst; loss=zeroone) = 0.9693
# (epoch,lwg...) = (2,0.072304815f0,25.057308f0,8.820978f0)
# 1 - test(lenet,dtrn; loss=zeroone) = 0.9771500000000002
# 1 - test(lenet,dtst; loss=zeroone) = 0.9775
# (epoch,lwg...) = (3,0.050180204f0,25.779385f0,9.264138f0)
# 1 - test(lenet,dtrn; loss=zeroone) = 0.9835666666666668
# 1 - test(lenet,dtst; loss=zeroone) = 0.982
