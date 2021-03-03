export MobileNetV3

import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Sequential, Op, Residual, SqueezeExcitation
using Knet.Ops20: pool
using Knet.Ops21: relu, hardswish, hardsigmoid
using Artifacts
# include("mobilenet_v1.jl") # (torch|keras)_mobilenet_preprocess, adaptive_avg_pool


"""
--DRAFT--

    MobileNetV3(; kwargs...)
    MobileNetV3(name::String; pretrained=true)

Return a MobileNet V3 model. The first call above returns a randomly initialized model, the
second loads a pretrained model. Pretrained models:

    name                           size  flops  top1  settings
    ----                           ----  -----  ----  --------
    mobilenet_v3_large_100_224_pt
    mobilenet_v3_small_100_224_pt


Keyword arguments:
* `width = 1`
* `resolution = 224`
* `input = 16`
* `output1 = 960`
* `output2 = 1280`
* `classes = 1000`
* `dropout = 0.2`
* `layout = mobilenet_v3_large_layout`
* `preprocess = torch_mobilenet_preprocess`

References:
* https://arxiv.org/abs/1905.02244

"""
function MobileNetV3(
    ;
    width = 1,
    resolution = 224,
    input = 16,
    output1 = 960,
    output2 = 1280,
    classes = 1000,
    dropout = 0.2,
    layout = mobilenet_v3_large_layout,
    preprocess = torch_mobilenet_preprocess,
)
    α(x) = round(Int, width*x)
    s = Sequential()
    push!(s, preprocess(resolution))
    push!(s, conv_bn(3, 3, 3, α(input); stride = 2, activation = hardswish))
    channels = input
    for l in layout
        stride, expand, squeeze = l.stride, α(l.expand), α(l.squeeze)
        for r in 1:l.repeat
            @assert channels == l.input
            push!(s, mobilenet_v3_block(α(channels), α(l.output); stride, expand, squeeze, l.kernel, l.activation))
            stride, channels = 1, l.output
        end
    end
    push!(s, conv_bn(1, 1, α(channels),  α(output1); activation=hardswish))
    push!(s, adaptive_avg_pool)
    push!(s, Linear(α(output1), α(output2); binit=zeros, activation=hardswish))
    push!(s, Linear(α(output2), classes; binit=zeros, dropout))
    return s
end    


function mobilenet_v3_block(input, output; expand=0, squeeze=0, kernel=3, stride=1, activation=relu)
    s = Sequential()
    channels = input
    if expand > 0
        push!(s, conv_bn(1, 1, channels, expand; activation))
        channels = expand
    end
    push!(s, conv_bn(kernel, kernel, 1, channels; activation, groups=channels, stride))
    if squeeze > 0
        push!(s, SqueezeExcitation(
            Conv(1, 1, channels, squeeze; binit=zeros, activation=relu),
            Conv(1, 1, squeeze, channels; binit=zeros, activation=hardsigmoid),
        ))
    end
    push!(s, conv_bn(1, 1, channels, output; activation=nothing))
    return input == output && stride == 1 ? Residual(s) : s
end


function conv_bn(w,h,x,y; activation=relu, groups=1, stride=1, bnupdate=0.01, bnepsilon=0.001)
    padding = (w-1)÷2
    normalization=BatchNorm(; update=bnupdate, epsilon=bnepsilon)
    Conv(w,h,x,y; normalization, groups, stride, padding, activation)
end


mobilenet_v3_large_layout = (
    # first input=16, last output through 160=>960 hardswish before pool, then linear 960=>1280=>1000
    (repeat=1, input=16, expand=0, squeeze=0, output=16, kernel=3, stride=1, activation=relu),
    (repeat=1, input=16, expand=64, squeeze=0, output=24, kernel=3, stride=2, activation=relu),
    (repeat=1, input=24, expand=72, squeeze=0, output=24, kernel=3, stride=1, activation=relu),
    (repeat=1, input=24, expand=72, squeeze=24, output=40, kernel=5, stride=2, activation=relu),
    (repeat=2, input=40, expand=120, squeeze=32, output=40, kernel=5, stride=1, activation=relu),
    (repeat=1, input=40, expand=240, squeeze=0, output=80, kernel=3, stride=2, activation=hardswish),
    (repeat=1, input=80, expand=200, squeeze=0, output=80, kernel=3, stride=1, activation=hardswish),
    (repeat=2, input=80, expand=184, squeeze=0, output=80, kernel=3, stride=1, activation=hardswish),
    (repeat=1, input=80, expand=480, squeeze=120, output=112, kernel=3, stride=1, activation=hardswish),
    (repeat=1, input=112, expand=672, squeeze=168, output=112, kernel=3, stride=1, activation=hardswish),
    (repeat=1, input=112, expand=672, squeeze=168, output=160, kernel=5, stride=2, activation=hardswish),
    (repeat=2, input=160, expand=960, squeeze=240, output=160, kernel=5, stride=1, activation=hardswish),
)


mobilenet_v3_small_layout = (
    # first input=16, last output through 96=>576 hardswish before pool, then linear 576=>1024=>1000
    (repeat=1, input=16, expand=0, squeeze=8, output=16, kernel=3, stride=2, activation=relu),
    (repeat=1, input=16, expand=72, squeeze=0, output=24, kernel=3, stride=2, activation=relu),
    (repeat=1, input=24, expand=88, squeeze=0, output=24, kernel=3, stride=1, activation=relu),
    (repeat=1, input=24, expand=96, squeeze=24, output=40, kernel=5, stride=2, activation=hardswish),
    (repeat=2, input=40, expand=240, squeeze=64, output=40, kernel=5, stride=1, activation=hardswish),
    (repeat=1, input=40, expand=120, squeeze=32, output=48, kernel=5, stride=1, activation=hardswish),
    (repeat=1, input=48, expand=144, squeeze=40, output=48, kernel=5, stride=1, activation=hardswish),
    (repeat=1, input=48, expand=288, squeeze=72, output=96, kernel=5, stride=2, activation=hardswish),
    (repeat=2, input=96, expand=576, squeeze=144, output=96, kernel=5, stride=1, activation=hardswish),
)


## Pretrained models

mobilenet_v3_models = Dict{String,NamedTuple}(
    "mobilenet_v3_large_100_224_pt" => (width=1, resolution=224, input=16, output1=960, output2=1280, classes=1000, dropout=0.2, layout=mobilenet_v3_large_layout, preprocess=torch_mobilenet_preprocess),
    "mobilenet_v3_small_100_224_pt" => (width=1, resolution=224, input=16, output1=576, output2=1024, classes=1000, dropout=0.2, layout=mobilenet_v3_small_layout, preprocess=torch_mobilenet_preprocess),
)

function MobileNetV3(s::String; pretrained=true)
    @assert haskey(mobilenet_v3_models, s)  "Please choose from known MobileNetV3 models:\n$(collect(keys(mobilenet_v3_models)))"
    kwargs = mobilenet_v3_models[s]
    model = MobileNetV3(; kwargs...)
    model(Knet.atype(zeros(Float32,224,224,3,1)))
    pretrained && setweights!(model, joinpath(@artifact_str(s), "$s.jld2"))
    return model
end
