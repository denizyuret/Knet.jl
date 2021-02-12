using PyCall, FileIO, ImageCore, ImageTransformations, CUDA
import Knet
using Knet.Layers21, Knet.Ops21
using Knet.Train20: param
using Knet.Ops20: pool, softmax
Knet.atype() = Array{Float32}
typename(p::PyObject) = pytypeof(p).__name__
tf = pyimport("tensorflow")
include("imagenet.jl")


function kerastest(img="fooval/ILSVRC2012_val_00000001.JPEG")
    pf = tf.keras.applications.MobileNet()
    jf = keras2knet(pf)
    px = imagenet_preprocess(img; normalization="tf", format="nhwc", atype=Array{Float32}) # 1,224,224,3
    jx = Knet.atype(permutedims(px,(2,3,4,1))) # 224,224,3,1
    @time py = pf(px).numpy()
    @time jy = jf(jx)
    isapprox(Array(jy), permutedims(py, (2,1)))
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
    x->begin #ZeroPadding2D
        w = oftype(x, reshape(Float32[0,0,0,1], 2, 2, 1, 1))
        y = conv(w, reshape(x, size(x,1), size(x,2), 1, :); padding=1)
        reshape(y, size(y,1), size(y,2), size(x,3), size(x,4))
    end
end


function keras_Conv2D(l, layers)
    # channels_last corresponds to inputs with shape (batch_size, height, width, channels) while 
    # channels_first corresponds to inputs with shape (batch_size, channels, height, width)
    w = param(l.weights[1].numpy())
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


function keras_DepthwiseConv2D(l, layers)
    w = l.weights[1].numpy()
    @assert size(w,4) == 1
    w = param(permutedims(w, (1,2,4,3))) ### (3,3,32,1) => (3,3,1,32)
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


function keras_ReLU(l, layers)
    max_value = (l.max_value[] === nothing ? Inf : l.max_value[])
    threshold = l.threshold[]
    negative_slope = l.negative_slope[]
    x->relu(x; max_value, threshold, negative_slope)
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
    x->begin #Reshape
        t = (l.target_shape..., size(x)[end])
        reshape(x, t)
    end
end


function keras_Dropout(l, layers)
    x->dropout(x, l.rate)
end


function keras_Activation(l, layers)
    @assert l.activation.__name__ == "softmax"  "Activation $(l.activation.__name__) not yet implemented."
    x->softmax(x; dims=1)
end


