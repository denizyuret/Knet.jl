export MobileNetV1

import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Sequential, Op
using Knet.Ops20: pool
using Knet.Ops21: relu
using Artifacts
# include("imagenet.jl") # imagenet_preprocess
# include("deepmap.jl")  # deepmap
# include("fileio.jl")   # setweights!


"""
--DRAFT--

    MobileNetV1(; kwargs...)
    MobileNetV1(name::String; pretrained=true)

Return a MobileNet V1 model. The first call above returns a randomly initialized model, the
second loads a pretrained model. Pretrained models:

    name                  size  flops  top1  settings
    ----                  ----  -----  ----  --------


Keyword arguments:
* `width = 1`
* `resolution = 224`
* `input = 32`
* `classes = 1000`
* `padding = 1`
* `bnupdate = 0.1`
* `bnepsilon = 1e-5`
* `preprocess = torch_mobilenet_preprocess`

References:
* https://arxiv.org/abs/1704.04861

"""
function MobileNetV1(
    ;
    width = 1,
    resolution = 224,
    input = 32,
    classes = 1000,
    padding = 1,                # torch:1 keras:((0,1),(0,1))
    bnupdate = 0.1,             # torch:0.1, keras.MobileNetV1:0.01, keras.MobileNetV2:0.001
    bnepsilon = 1e-5,           # torch:1e-5, keras:0.001
    dropout = 0.001,
    preprocess = torch_mobilenet_preprocess, # (keras|torch)_mobilenet_preprocess
)
    α(x) = round(Int, width*x)
    s = Sequential()
    push!(s, preprocess(resolution))
    push!(s, conv_bn_relu6(3, 3, 3, α(input); stride = 2, padding, bnupdate, bnepsilon))
    channels = input
    for l in mobilenet_v1_layout
        stride, output = l.stride, l.output
        for r in 1:l.repeat
            push!(s, mobilenet_v1_block(α(channels), α(output); stride, padding, bnupdate, bnepsilon))
            channels = l.output
            stride = 1
        end
    end
    push!(s, adaptive_avg_pool)
    push!(s, Linear(α(channels), classes; binit=zeros, dropout))
    return s
end    


const mobilenet_v1_layout = (
    (repeat=1, output=64, stride=1),
    (repeat=2, output=128, stride=2),
    (repeat=2, output=256, stride=2),
    (repeat=6, output=512, stride=2),
    (repeat=2, output=1024, stride=2),
)


function mobilenet_v1_block(x, y; stride = 1, o...)
    Sequential(
        conv_bn_relu6(3, 3, 1, x; groups=x, stride, o...),
        conv_bn_relu6(1, 1, x, y; o...),
    )
end


function conv_bn_relu6(w,h,x,y; groups = 1, stride = 1, padding = 1, bnupdate=0.1, bnepsilon=1e-5, activation=relu6)
    padding = (w == 1 ? 0 : stride == 1 ? 1 : padding)
    normalization=BatchNorm(; update=bnupdate, epsilon=bnepsilon)
    Conv(w,h,x,y; normalization, groups, stride, padding, activation)
end


function adaptive_avg_pool(x)
    y = pool(x; mode=1, window=size(x)[1:end-2])
    reshape(y, size(y,3), size(y,4))
end


const relu6 = Op(relu; max_value=6)
torch_mobilenet_preprocess(resolution) = Op(imagenet_preprocess; normalization="torch", format="whcn", resolution)
keras_mobilenet_preprocess(resolution) = Op(imagenet_preprocess; normalization="tf", format="whcn", resolution)


## Pretrained models

mobilenet_v1_models = Dict{String,NamedTuple}(
    "mobilenet_v1_100_224_tf" => (width=1, resolution=224, input=32, classes=1000, padding=((0,1),(0,1)), bnupdate=0.01, bnepsilon=0.001, preprocess=keras_mobilenet_preprocess),
)

function MobileNetV1(s::String; pretrained=true)
    @assert haskey(mobilenet_v1_models, s)  "Please choose from known MobileNetV1 models:\n$(collect(keys(mobilenet_v1_models)))"
    kwargs = mobilenet_v1_models[s]
    model = MobileNetV1(; kwargs...)
    model(Knet.atype(zeros(Float32,224,224,3,1)))
    pretrained && setweights!(model, joinpath(@artifact_str(s), "$s.jld2"))
    return model
end
