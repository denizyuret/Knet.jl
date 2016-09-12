# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# 4-D convolution test with the LeNet model.

isdefined(:MNIST) || (load_only=true;include("mnist.jl"))

module LeNet
using Knet,ArgParse,AutoGrad,CUDArt,CUDNN
using Main.MNIST: minibatch, xtrn, ytrn, xtst, ytst
gpu(true)


function main(args=ARGS)
    info("Testing LeNet (convolutional net) on MNIST")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--batchsize"; arg_type=Int; default=100)
        ("--lr"; arg_type=Float64; default=0.1)
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--epochs"; arg_type=Int; default=3)
        #TODO: ("--gcheck"; arg_type=Int; default=0), --atype, --winit
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])

    global dtrn = minibatch4(xtrn, ytrn, o[:batchsize])
    global dtst = minibatch4(xtst, ytst, o[:batchsize])
    global w = weights()

    if o[:fast]
        @time train(w, dtrn; lr=o[:lr], epochs=o[:epochs])
    else
        println((:epoch,0,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
        @time for epoch=1:o[:epochs]
            train(w, dtrn; lr=o[:lr], epochs=1)
            println((:epoch,epoch,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
        end
    end
    return w
end

function train(w, data; lr=.1, epochs=20, nxy=0)
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

function predict(w,x0)                       # 28,28,1,100
    x1 = pool4(relu(conv4(w[1],x0) .+ w[2])) # 12,12,20,100
    x2 = pool4(relu(conv4(w[3],x1) .+ w[4])) # 4,4,50,100
    x2a = reshape(x2, (800,100))             # 800,100
    x3 = relu(w[5]*x2a .+ w[6])              # 500,100
    x4 = w[7]*x3 .+ w[8]                     # 10,100
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = logp(ypred)  # ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function weights(;ftype=Float32,atype=KnetArray)
    w = Array(Any,8)
    w[1] = xavier(Float32,5,5,1,20)
    w[2] = zeros(Float32,1,1,20,1)
    w[3] = xavier(Float32,5,5,20,50)
    w[4] = zeros(Float32,1,1,50,1)
    w[5] = xavier(Float32,500,800)
    w[6] = zeros(Float32,500,1)
    w[7] = xavier(Float32,10,500)
    w[8] = zeros(Float32,10,1)
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

function accuracy(w, dtst; nxy=0)
    ncorrect = ninstance = nloss = 0
    for (x, ygold) in dtst
        ypred = predict(w, x)
        ynorm = ypred .- log(sum(exp(ypred),1))
        nloss += -sum(ygold .* ynorm)
        ncorrect += sum((ypred .== maximum(ypred,1)) .* (ygold .== maximum(ygold,1)))
        ninstance += size(ygold,2)
    end
    return (ncorrect/ninstance, nloss/ninstance)
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


# Define some new primitives: conv4 and pool4

function conv4{T}(w::KnetArray{T},x::KnetArray{T}; o...)
    cx = CudaArray(x)
    cw = CudaArray(w)
    ydims = cudnnGetConvolutionNdForwardOutputDim(cx,cw; o...)
    y = similar(x, ydims)
    cy = CudaArray(y)
    cudnnConvolutionForward(cx, cw, cy; o...)
    return y
end

function conv4x{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; o...)
    dx = similar(x)
    cw = CudaArray(w)
    cdx = CudaArray(dx)
    cdy = CudaArray(dy)
    cudnnConvolutionBackwardData(cw,cdy,cdx; o...)
    return dx
end

function conv4w{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; o...)
    dw = similar(w)
    cx = CudaArray(x)
    cdy = CudaArray(dy)
    cdw = CudaArray(dw)
    cudnnConvolutionBackwardFilter(cx,cdy,cdw; o...)
    return dw
end

@primitive  conv4(w,x; o...),dy  conv4w(w,x,dy;o...)  conv4x(w,x,dy;o...)
@zerograd conv4x(w,x,dy;o...)
@zerograd conv4w(w,x,dy;o...)

function pool4{T}(x::KnetArray{T}; o...)
    pd = CUDNN.PD(ndims=ndims(x), o...)
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
@zerograd pool4x(x,y,dy;o...)

# CUDNN supports CudaArrays, here is a hack until we implement KnetArray support

using Knet: KnetPtr
Base.convert(::Type{CudaPtr}, p::KnetPtr)=CudaPtr(p.ptr)
Base.convert{T,N}(::Type{CudaArray}, x::KnetArray{T,N})=CudaArray{T,N}(CudaPtr(x.ptr), size(x), x.dev)


!isinteractive() && main(ARGS) # !isdefined(Core.Main,:load_only) && main(ARGS)

end # module
