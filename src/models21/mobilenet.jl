export MobileNet

import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Sequential, Op
using Knet.Ops20: pool
using Knet.Ops21: relu


"""
--DRAFT--

    MobileNet(; kwargs...)
    MobileNet(name::String; pretrained=true)

Return a MobileNet model. Pretrained models:

    name                  size  flops  top1  settings
    ----                  ----  -----  ----  --------
    resnet50               98M   1.00  .753
    mobilenet_v1_100_224   17M   0.27  .704  (block=DWConv, layout=mobilenet_v1_layout, output=1024)
    mobilenet_v2_100_224   14M   0.44  .713  (block=MBConv, layout=mobilenet_v2_layout, output=1280)

Keyword arguments:
* `layout = mobilenet_v2_layout`
* `block = MBConv`
* `preprocess = torch_mobilenet_preprocess`
* `width = 1`
* `resolution = 224`
* `input = 32`
* `output = 1280`
* `classes = 1000`
* `padding = 1`

References:
* https://arxiv.org/abs/1704.04861
* https://arxiv.org/abs/1801.04381
* https://arxiv.org/abs/1905.02244

"""
function MobileNet(
    ;
    width = 1,
    resolution = 224,
    input = 32,
    output = 1280,
    classes = 1000,
    block = MBConv,
    layout = mobilenet_v2_layout,
    preprocess = torch_mobilenet_preprocess, # use keras_mobilenet_preprocess for keras models
    padding = 1,                # ((0,1),(0,1)) for keras models
    bnupdate = 0.1,             # torch:0.1, keras.MobileNetV1:0.01, keras.MobileNetV2:0.001
    bnepsilon = 1e-5,           # torch:1e-5, keras:0.001
)
    α(x) = round(Int, width*x)
    s = Sequential()
    push!(s, preprocess(resolution))
    push!(s, ConvBN6(3, 3, 3, α(input); stride = 2, padding, bnupdate, bnepsilon))
    channels = input
    for (repeat, outchannels, stride, expansion) in layout
        for r in 1:repeat
            push!(s, block(α(channels), α(outchannels); stride, expansion, padding, bnupdate, bnepsilon))
            channels = outchannels
            stride = 1
        end
    end
    channels != output && push!(s, ConvBN6(1, 1, α(channels),  α(output); padding, bnupdate, bnepsilon))
    push!(s, adaptive_avg_pool)
    push!(s, Linear(α(output), classes; binit=zeros))
    return s
end    


function MBConv(x, y; stride = 1, expansion = 6, o...)
    b = expansion * x
    s = Sequential(
        ConvBN6(3, 3, 1, b; groups=b, stride, o...),
        ConvBN6(1, 1, b, y; activation=nothing, o...),
    )
    b != x && pushfirst!(s, ConvBN6(1, 1, x, b; o...))
    x == y ? Residual(s) : s
end


function DWConv(x, y; stride = 1, expansion = 1, o...)
    Sequential(
        ConvBN6(3, 3, 1, x; groups=x, stride, o...),
        ConvBN6(1, 1, x, y; o...),
    )
end


function ConvBN6(w,h,x,y; groups = 1, stride = 1, padding = 1, bnupdate=0.1, bnepsilon=1e-5,
                 activation=Op(relu; max_value=6))
    padding = (w == 1 ? 0 : stride == 1 ? 1 : padding)
    normalization=BatchNorm(; update=bnupdate, epsilon=bnepsilon)
    Conv(w,h,x,y; normalization, groups, stride, padding, activation)
end


function adaptive_avg_pool(x)
    y = pool(x; mode=1, window=size(x)[1:end-2])
    reshape(y, size(y,3), size(y,4))
end


torch_mobilenet_preprocess(resolution) = Op(imagenet_preprocess; normalization="torch", format="whcn", resolution)
keras_mobilenet_preprocess(resolution) = Op(imagenet_preprocess; normalization="tf", format="whcn", resolution)


## Pretrained models

function MobileNet(s::String; pretrained=true)
    @assert haskey(mobilenetmodels, s)  "Please choose from known MobileNet models:\n$(collect(keys(mobilenetmodels)))"
    kwargs = mobilenetmodels[s]
    model = MobileNet(; kwargs...)
    model(Knet.atype(zeros(Float32,224,224,3,1)))
    pretrained && setweights!(model, joinpath(@artifact_str(s), "$s.jld2"))
    return model
end


# repeat, output, stride, expansion
const mobilenet_v1_layout = (
    (1,  64, 1, 1),
    (2, 128, 2, 1),
    (2, 256, 2, 1),
    (6, 512, 2, 1),
    (2, 1024, 2, 1),
)

const mobilenet_v2_layout = (
    (1, 16, 1, 1),
    (2, 24, 2, 6),
    (3, 32, 2, 6),
    (4, 64, 2, 6),
    (3, 96, 1, 6),
    (3, 160, 2, 6),
    (1, 320, 1, 6),
)


# mobilenet models from keras.applications and torchvision.models
mobilenetmodels = Dict{String,NamedTuple}(
    "mobilenet_v1_100_224_tf" => (block=DWConv, layout=mobilenet_v1_layout, output=1024, preprocess=keras_mobilenet_preprocess, padding=((0,1),(0,1)), bnupdate=0.01, bnepsilon=0.001),
    "mobilenet_v2_100_224_tf" => (block=MBConv, layout=mobilenet_v2_layout, output=1280, preprocess=keras_mobilenet_preprocess, padding=((0,1),(0,1)), bnupdate=0.001, bnepsilon=0.001),
    "mobilenet_v2_100_224_pt" => (block=MBConv, layout=mobilenet_v2_layout, output=1280, preprocess=torch_mobilenet_preprocess, padding=1, bnupdate=0.1, bnepsilon=1e-5),
)

nothing
