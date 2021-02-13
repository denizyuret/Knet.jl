using PyCall, FileIO, ImageCore, ImageTransformations, CUDA
import Knet
using Knet.Layers21, Knet.Ops21
using Knet.Train20: param
using Knet.Ops20: pool, softmax
typename(p::PyObject) = pytypeof(p).__name__
tf = pyimport("tensorflow")
include("imagenet.jl")


function kerastest(img="fooval/ILSVRC2012_val_00000001.JPEG")
    global pf, jf, px, jx
    pf = tf.keras.applications.MobileNet()
    jf = keras2knet(pf)
    px = imagenet_preprocess(img; normalization="tf", format="nhwc", atype=Array{Float32}) # 1,224,224,3
    jx = Knet.atype(permutedims(px,(3,2,4,1))) # 224,224,3,1
    @time py = pf(px).numpy() |> vec
    @time jy = jf(jx) |> vec
    @show jy ≈ py
    idx = sortperm(py, rev=true)
    [ idx py[idx] imagenet_labels()[idx] ]
end


# Model translator
function keras2knet(p::PyObject; xtest = nothing, nlayers = -1)
    layers = nlayers >= 1 ? p.layers[1:nlayers] : copy(p.layers)
    model = Sequential(; name=p.name)
    while !isempty(layers)
        layer1 = popfirst!(layers)
        ltype = typename(layer1)
        # This could also pop one or more layers:
        layer2 = eval(Meta.parse("keras_$ltype"))(layer1, layers)
        layer2 !== nothing && push!(model, layer2)
    end
    if xtest !== nothing # x = rand(Float32,1,224,224,3)
        y1 = p.predict(xtest)
        y2 = model(xtest)
        @assert vec(Array(y1)) ≈ vec(Array(y2))
    end
    return model
end


keras_InputLayer(l, layers) = nothing


function keras_ZeroPadding2D(l, layers)
    @assert l.padding == ((0, 1), (0, 1)) "padding=$(l.padding) not implemented yet"
    ZeroPad01()
end


function keras_Conv2D(l, layers)
    # channels_last corresponds to inputs with shape nhwc (default)
    # channels_first corresponds to inputs with shape nchw
    # filters in channels_last are in hwxy order
    # knet default is whcn for images and whxy for filters
    w = param(permutedims(l.weights[1].numpy(),(2,1,3,4)))
    bias = (l.bias === nothing ? nothing : param(reshape(l.bias.numpy(), (1,1,:,1))))
    normalization, activation = keras_normact(layers)
    channelmajor = false
    crosscorrelation = true
    dilation = l.dilation_rate
    groups = l.groups
    padding = (l.padding == "valid" ? 0 : l.padding == "same" ? (size(w)[1:2] .- 1) .÷ 2 : l.padding)
    stride = l.strides
    Conv(w; activation, normalization, bias, channelmajor, crosscorrelation, dilation, groups, padding, stride)
end


function keras_DepthwiseConv2D(l, layers)
    w = l.weights[1].numpy()
    @assert size(w,4) == 1
    w = param(permutedims(w, (2,1,4,3))) ### (3,3,32,1) => (3,3,1,32)
    bias = (l.bias === nothing ? nothing : param(reshape(l.bias.numpy(), (1,1,:,1))))
    normalization, activation = keras_normact(layers)
    channelmajor = false
    crosscorrelation = true
    dilation = l.dilation_rate
    groups = size(w,4) ###
    padding = (l.padding == "valid" ? 0 : l.padding == "same" ? (size(w)[1:2] .- 1) .÷ 2 : l.padding)
    stride = l.strides
    Conv(w; activation, normalization, bias, channelmajor, crosscorrelation, dilation, groups, padding, stride)
end


function keras_normact(layers)
    if !isempty(layers) && typename(layers[1]) == "BatchNormalization"
        normalization = keras_BatchNormalization(popfirst!(layers), layers)
    else
        normalization = nothing
    end
    if !isempty(layers) && typename(layers[1]) == "ReLU"
        activation = keras_ReLU(popfirst!(layers), layers)
    else
        activation = nothing
    end
    normalization, activation
end    


function keras_ReLU(l, layers)
    max_value = (l.max_value[] === nothing ? Inf : l.max_value[])
    threshold = l.threshold[]
    negative_slope = l.negative_slope[]
    if threshold == negative_slope == 0
        Op(relu; max_value)
    else
        Op(relu; max_value, threshold, negative_slope)
    end
end


function keras_BatchNormalization(l, layers)
    bparam(x) = param(reshape(x.numpy(), (1,1,:,1)))
    mean = bparam(l.moving_mean).value
    var = bparam(l.moving_variance).value
    bias = bparam(l.beta)
    scale = bparam(l.gamma)
    epsilon = l.epsilon
    update = 1-l.momentum
    BatchNorm(; mean, var, bias, scale, epsilon, update)
end


function keras_GlobalAveragePooling2D(l, layers)
    x->begin #GlobalAveragePooling2D
        y = pool(x; mode=1, window=size(x)[1:2])
        reshape(y, size(y,3), size(y,4))
    end
end


function keras_Reshape(l, layers)
    Op(reshape, (l.target_shape..., :))
end


function keras_Dropout(l, layers)
    Op(dropout, l.rate)
end


function keras_Activation(l, layers)
    @assert l.activation.__name__ == "softmax"  "Activation $(l.activation.__name__) not yet implemented."
    Op(softmax; dims=1)
end

mutable struct ZeroPad01; w; end

ZeroPad01() = ZeroPad01(nothing)

function (z::ZeroPad01)(x)
    w,h,c,n = size(x)
    if typeof(z.w) != typeof(x) || size(z.w) != (2,2,1,c)
	z.w = oftype(x, zeros(Float32, 2, 2, 1, c))
	z.w[2,2,1,:] .= 1
    end
    conv(z.w, x; padding = 1, groups = c)
end

Base.show(io::IO, ::MIME"text/plain", o::ZeroPad01) = show(io, o)
Base.show(io::IO, o::ZeroPad01) = print(io, ZeroPad01)
