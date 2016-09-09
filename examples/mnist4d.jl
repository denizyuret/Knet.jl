# TODO:
# control memory for backward pass

# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# 4-D convolution test

isdefined(:MNIST) || include("mnist.jl")

# module MNIST4D
using Knet,ArgParse,AutoGrad,CUDArt,CUDNN
using Main.MNIST: minibatch, xtrn, ytrn, xtst, ytst
gpu(true)

function predict(w,x0)      # LeNet model input: 28,28,1,100
    x1 = pool4(max(0, conv4(w[1],x0) .+ w[2])) # 12,12,20,100: (4608000,3),(1152000,1)
    x2 = pool4(max(0, conv4(w[3],x1) .+ w[4])) # 4,4,50,100: (1280000,3),(320000,1)
    x3 = max(0, w[5]*x2 .+ w[6])               # 500,100: (200000,3)
    x4 = w[7]*x3 .+ w[8]                       # 10,100: (4000,2)
end

function loss(w,x,ygold)
    ypred = predict(w,x)        # 10,100
    ynorm = ypred .- log(sum(exp(ypred),1)) # (4000,2)(400,2)
    -sum(ygold .* ynorm) / size(ygold,2)    # (4000,1)
end

lossgradient = grad(loss)

function weights(;ftype=Float32,atype=KnetArray,winit=0.1) # TODO: xavier
    w = Array(Any,8)
    w[1] = randn(Float32,5,5,1,20)*winit
    w[2] = zeros(Float32,1,1,20,1)
    w[3] = randn(Float32,5,5,20,50)*winit
    w[4] = zeros(Float32,1,1,50,1)
    w[5] = randn(Float32,500,800)*winit
    w[6] = zeros(Float32,500,1)
    w[7] = randn(Float32,10,500)*winit
    w[8] = zeros(Float32,10,1)
    return map(a->convert(atype,a), w)
end

function minibatch4(x, y, batchsize; atype=KnetArray{Float32})
    global data = minibatch(x,y,batchsize; atype=atype)
    for i=1:length(data)
        (x,y) = data[i]
        data[i] = (reshape(x, (28,28,1,batchsize)), y)
    end
    return data
end

function accuracy(w, dtst; nxy=0)
    ncorrect = ninstance = 0
    for (x, ygold) in dtst
        ypred = predict(w, x)
        ncorrect += sum((ypred .== maximum(ypred,1)) .* (ygold .== maximum(ygold,1)))
        ninstance += size(ygold,2)
        (nxy+=1)%100==0 && gc()
    end
    knetgc()
    return ncorrect/ninstance
end

function train(w, data; lr=.1, epochs=20, nxy=0)
    for epoch=1:epochs
        for (x,y) in data
            global g = lossgradient(w, x, y)
            for i in 1:length(w)
                w[i] -= lr * g[i]
            end
            (nxy+=1)%100==0 && gc()
        end
    end
    knetgc()
    return w
end

function main(args=ARGS)
    info("Testing lenet (convolutional net) on MNIST")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--batchsize"; arg_type=Int; default=100)
        ("--lr"; arg_type=Float64; default=0.0)
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

# CUDNN supports CudaArrays, here is a hack until we implement KnetArray support

using Knet: KnetPtr
import Base: convert, randn
convert(::Type{CudaPtr}, p::KnetPtr)=CudaPtr(p.ptr)
convert{T,N}(::Type{CudaArray}, x::KnetArray{T,N})=CudaArray{T,N}(CudaPtr(x.ptr), size(x), x.dev)

# randn does not support type?
randn(T::Type, dims::Dims) = convert(Array{T}, randn(dims))
randn(T::Type, d1::Integer, dims::Integer...) = randn(T, convert(Tuple{Vararg{Int}}, (d1,dims...)))

# deal with matmul of 4D arrays
mat{T}(B::KnetArray{T,2})=B
mat(B::KnetArray)=(b2=size(B,ndims(B));b1=div(length(B),b2);reshape(B,(b1,b2)))
Base.(:*)(A::KnetArray,B::KnetArray)=(mat(A)*mat(B))
Base.A_mul_Bc(A::KnetArray, B::KnetArray)=A_mul_Bc(mat(A),mat(B))
Base.Ac_mul_B(A::KnetArray, B::KnetArray)=Ac_mul_B(mat(A),mat(B))


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

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

# end # module


