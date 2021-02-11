using PyCall
using Knet.Layers21, Knet.Ops21
using Knet.Train20: param
using Knet.Ops20: pool, softmax
typename(p::PyObject) = pytypeof(p).__name__


function keras2knet(p::PyObject; xtest = nothing, nlayers = -1)
    layers = nlayers >= 1 ? p.layers[1:nlayers] : copy(p.layers)
    model = Sequential(; name=p.name)
    while !isempty(layers)
        @show ltype = typename(layers[1])
        # This should also pop one or more layers:
        layer21 = eval(Meta.parse("$(ltype)_keras"))(layers)
        layer21 !== nothing && push!(model, layer21)
    end
    if xtest !== nothing # x = rand(Float32,1,224,224,3)
        y1 = p.predict(xtest)
        y2 = model(xtest)
        @assert vec(Array(y1)) ≈ vec(Array(y2))
    end
    return model
end


function InputLayer_keras(layers)
    l = popfirst!(layers)
    nothing
end


function ZeroPadding2D_keras(layers)
    l = popfirst!(layers)
    @assert typename(l) == "ZeroPadding2D"
    @assert l.padding == ((0, 1), (0, 1))
    x->begin
        w = oftype(x, reshape(Float32[0,0,0,1], 2, 2, 1, 1))
        y = conv(w, reshape(x, size(x,1), size(x,2), 1, :); padding=1)
        reshape(y, size(y,1), size(y,2), size(x,3), size(x,4))
    end
end


function Conv2D_keras(layers)
    # channels_last corresponds to inputs with shape (batch_size, height, width, channels) while 
    # channels_first corresponds to inputs with shape (batch_size, channels, height, width)
    l = popfirst!(layers)
    w = l.weights[1].numpy() |> param
    bias = (l.bias === nothing ? nothing : param(reshape(l.bias.numpy(), (1,1,:,1))))
    normalization = (!isempty(layers) && typename(layers[1]) == "BatchNormalization" ? BatchNormalization_keras(layers) : nothing)
    activation = (!isempty(layers) && typename(layers[1]) == "ReLU" ? ReLU_keras(layers) : nothing)
    channelmajor = false
    crosscorrelation = true
    dilation = l.dilation_rate
    groups = l.groups
    padding = (l.padding == "valid" ? 0 : l.padding == "same" ? (size(w)[1:2] .- 1) .÷ 2 : l.padding)
    stride = l.strides
    Conv(w; activation, normalization, bias, channelmajor, crosscorrelation, dilation, groups, padding, stride)
end


function DepthwiseConv2D_keras(layers)
    l = popfirst!(layers)
    w = l.weights[1].numpy()
    @assert size(w,4) == 1
    w = param(permutedims(w, (1,2,4,3))) ###
    bias = (l.bias === nothing ? nothing : param(reshape(l.bias.numpy(), (1,1,:,1))))
    normalization = (!isempty(layers) && typename(layers[1]) == "BatchNormalization" ? BatchNormalization_keras(layers) : nothing)
    activation = (!isempty(layers) && typename(layers[1]) == "ReLU" ? ReLU_keras(layers) : nothing)
    channelmajor = false
    crosscorrelation = true
    dilation = l.dilation_rate
    groups = size(w,4) ###
    padding = (l.padding == "valid" ? 0 : l.padding == "same" ? (size(w)[1:2] .- 1) .÷ 2 : l.padding)
    stride = l.strides
    Conv(w; activation, normalization, bias, channelmajor, crosscorrelation, dilation, groups, padding, stride)
end


function ReLU_keras(layers)
    l = popfirst!(layers)
    @assert typename(l) == "ReLU"
    x->(x >= l.max_value[] ? l.max_value[] :
        x >= l.threshold[] ? x :
        l.negative_slope[] * (x - l.threshold[]))
end


function BatchNormalization_keras(layers)
    bparam(x) = param(reshape(x.numpy(), (1,1,:,1)))
    b = popfirst!(layers)
    @assert typename(b) == "BatchNormalization"
    mean = bparam(b.moving_mean).value
    var = bparam(b.moving_variance).value
    bias = bparam(b.beta)
    scale = bparam(b.gamma)
    epsilon = b.epsilon
    update = 1-b.momentum
    BatchNorm(; mean, var, bias, scale, epsilon, update)
end


function GlobalAveragePooling2D_keras(layers)
    l = popfirst!(layers)
    @assert typename(l) == "GlobalAveragePooling2D"
    x->begin
        y = pool(x; mode=1, window=size(x)[1:2])
        reshape(y, size(y,3), size(y,4))
    end
end


function Reshape_keras(layers)
    l = popfirst!(layers)
    @assert typename(l) == "Reshape"
    x->begin
        t = (l.target_shape..., size(x)[end])
        reshape(x, t)
    end
end


function Dropout_keras(layers)
    l = popfirst!(layers)
    @assert typename(l) == "Dropout"
    x->dropout(x, l.rate)
end


function Activation_keras(layers)
    l = popfirst!(layers)
    @assert typename(l) == "Activation"
    @assert l.activation.__name__ == "softmax"
    x->softmax(x; dims=1)
end
