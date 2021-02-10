using PyCall
using Knet.Layers21, Knet.Ops21
using Knet.Train20: param
typename(p::PyObject) = pytypeof(p).__name__


function keras2knet(p::PyObject; xtest = nothing)
    layers = copy(p.layers)
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
    @assert typename(layers[1]) == "Conv2D"
    nothing
end


function Conv2D_keras(layers)
    # channels_last corresponds to inputs with shape (batch_size, height, width, channels) while 
    # channels_first corresponds to inputs with shape (batch_size, channels, height, width)
    l = popfirst!(layers)
    w = l.weights[1].numpy() |> param
    bias = (l.bias === nothing ? nothing : param(reshape(l.bias.numpy(), (1,1,:,1))))
    normalization = (typename(layers[1]) == "BatchNormalization" ? BatchNormalization_keras(layers) : nothing)
    activation = (typename(layers[1]) == "ReLU" ? ReLU_keras(layers) : nothing)
    channelmajor = false
    crosscorrelation = true
    dilation = l.dilation_rate
    groups = l.groups
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
    l = popfirst!(layers)
    @assert typename(l) == "BatchNormalization"
    mean = bparam(b.moving_mean).value
    var = bparam(b.moving_variance).value
    bias = bparam(b.beta)
    scale = bparam(b.gamma)
    epsilon = b.epsilon
    update = 1-b.momentum
    BatchNorm(; mean, var, bias, scale, epsilon, update)
end
