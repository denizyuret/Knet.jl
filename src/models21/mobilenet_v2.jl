export MobileNetV2

import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Sequential, Residual, Op
using Knet.Ops20: pool
using Knet.Ops21: relu
using Artifacts
# include("mobilenet_v1.jl") # conv_bn_relu6, (torch|keras)_mobilenet_preprocess, adaptive_avg_pool

"""
--DRAFT--

    MobileNetV2(; kwargs...)
    MobileNetV2(name::String; pretrained=true)

Return a MobileNet V2 model. The first call above returns a randomly initialized model, the
second loads a pretrained model. Pretrained models:

    name                     size  flops  top1  settings
    ----                     ----  -----  ----  --------
    mobilenet_v2_100_224_tf
    mobilenet_v2_100_224_pt

Keyword arguments:
* `width = 1`
* `resolution = 224`
* `input = 32`
* `output = 1280`
* `classes = 1000`
* `padding = 1`
* `bnupdate = 0.1`
* `bnepsilon = 1e-5`
* `preprocess = torch_mobilenet_preprocess`

References:
* https://arxiv.org/abs/1801.04381

"""
function MobileNetV2(
    ;
    width = 1,
    resolution = 224,
    input = 32,
    output = 1280,
    classes = 1000,
    padding = 1,                # ((0,1),(0,1)) for keras models
    bnupdate = 0.1,             # torch:0.1, keras.MobileNetV1:0.01, keras.MobileNetV2:0.001
    bnepsilon = 1e-5,           # torch:1e-5, keras:0.001
    dropout = 0.2,
    preprocess = torch_mobilenet_preprocess, # use keras_mobilenet_preprocess for keras models
)
    α(x) = round(Int, width*x)
    s = Sequential()
    push!(s, preprocess(resolution))
    push!(s, conv_bn_relu6(3, 3, 3, α(input); stride = 2, padding, bnupdate, bnepsilon))
    channels = input
    for l in mobilenet_v2_layout
        stride = l.stride
        for r in 1:l.repeat
            push!(s, mobilenet_v2_block(α(channels), α(l.output); l.expansion, stride, padding, bnupdate, bnepsilon))
            channels = l.output
            stride = 1
        end
    end
    push!(s, conv_bn_relu6(1, 1, α(channels),  α(output); padding, bnupdate, bnepsilon))
    push!(s, adaptive_avg_pool)
    push!(s, Linear(α(output), classes; binit=zeros, dropout))
    return s
end    


const mobilenet_v2_layout = (
    (repeat=1, output=16, stride=1, expansion=1),
    (repeat=2, output=24, stride=2, expansion=6),
    (repeat=3, output=32, stride=2, expansion=6),
    (repeat=4, output=64, stride=2, expansion=6),
    (repeat=3, output=96, stride=1, expansion=6),
    (repeat=3, output=160, stride=2, expansion=6),
    (repeat=1, output=320, stride=1, expansion=6),
)


function mobilenet_v2_block(x, y; stride = 1, expansion = 6, o...)
    b = expansion * x
    s = Sequential()
    b != x && push!(s, conv_bn_relu6(1, 1, x, b; o...))
    push!(s, conv_bn_relu6(3, 3, 1, b; groups=b, stride, o...))
    push!(s, conv_bn_relu6(1, 1, b, y; activation=nothing, o...))
    x == y ? Residual(s) : s
end


## Pretrained models

mobilenet_v2_models = Dict{String,NamedTuple}(
    "mobilenet_v2_100_224_tf" => (width=1, resolution=224, input=32, output=1280, classes=1000, padding=((0,1),(0,1)), bnupdate=0.001, bnepsilon=0.001, preprocess=keras_mobilenet_preprocess),
    "mobilenet_v2_100_224_pt" => (width=1, resolution=224, input=32, output=1280, classes=1000, padding=1, bnupdate=0.1, bnepsilon=1e-5, preprocess=torch_mobilenet_preprocess),
)

function MobileNetV2(s::String; pretrained=true)
    @assert haskey(mobilenet_v2_models, s)  "Please choose from known MobileNetV2 models:\n$(collect(keys(mobilenet_v2_models)))"
    kwargs = mobilenet_v2_models[s]
    model = MobileNetV2(; kwargs...)
    model(Knet.atype(zeros(Float32,224,224,3,1)))
    pretrained && setweights!(model, joinpath(@artifact_str(s), "$s.jld2"))
    return model
end
