export MobileNet

import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Block, Op, Add, Mul
using Knet.Ops21: relu, hardswish, hardsigmoid, pool, mean, reshape2d
using Artifacts


"""
--DRAFT--

    MobileNet(; kwargs...)
    MobileNet(name::String; pretrained=true)

Return a MobileNet model. The first call above returns a randomly initialized model, the
second loads a pretrained model. The models satisfy the indexing interface, e.g. model[1] is
the preprocessing layer and model[1:end-1] omits the imagenet classification top.

Pretrained models:

    name                           top1   ref1   size  time  settings
    ----                           -----  -----  ----  ----  --------
    mobilenet_v1_100_224_tf        70.59  70.6   17M
    mobilenet_v2_100_224_tf        71.88  71.8   14M
    mobilenet_v2_100_224_pt        70.76  71.88  14M
    mobilenet_v3_small_100_224_pt  66.78  67.67  10M
    mobilenet_v3_large_100_224_pt  73.48  74.04  22M

Keyword arguments:

* `width = 1`
* `resolution = 224`
* `input = 16`
* `output = (960,1280)`
* `classes = 1000`
* `activation = hardswish`
* `tfpadding = false`
* `bnupdate = 0.01`
* `bnepsilon = 0.001`
* `dropout = 0.2`
* `block = mobilenet_v3_block`
* `layout = mobilenet_v3_large_layout`
* `preprocess = torch_mobilenet_preprocess`

References:
* https://arxiv.org/abs/1704.04861
* https://arxiv.org/abs/1801.04381
* https://arxiv.org/abs/1905.02244
* https://pytorch.org/vision/master/models.html
* https://keras.io/api/applications

"""
function MobileNet(
    ;
    width = 1,
    resolution = 224,
    input = 16,
    output = (960,1280),
    classes = 1000,
    activation = hardswish,      # TODO: document kwargs
    tfpadding = false,           # torch:1 keras:((0,1),(0,1)); torch:2, keras:((1,2),(1,2))
    bnupdate = 0.01,             # torchV2:0.1, torchV3:0.01, kerasV1:0.01, kerasV2:0.001
    bnepsilon = 0.001,           # torchV2:1e-5, torchV3:0.001, keras:0.001
    dropout = 0.2,               # kerasV1:0.001, others:0.2
    block = mobilenet_v3_block,
    layout = mobilenet_v3_large_layout,
    preprocess = torch_mobilenet_preprocess, # (keras|torch)_mobilenet_preprocess
)
    global _mobilenet_config = (; tfpadding, bnupdate, bnepsilon)
    α(x) = round(Int, width*x)
    s = Block()
    push!(s, preprocess(resolution))
    push!(s, mobilenet_conv_bn(3, 3, 3, α(input); stride=2, activation))
    channels = input
    for l in layout
        for r in 1:l.repeat
            push!(s, block(channels, l.output; layout=l, repeat=r, width))
            channels = l.output
        end
    end
    top = Block()
    if isempty(output)
        push!(top, Op(pool; op=mean, window=typemax(Int)))
        push!(top, reshape2d)
        push!(top, Linear(α(channels), classes; binit=zeros, dropout))
    else
        push!(top, mobilenet_conv_bn(1, 1, α(channels), α(output[1]); activation))
        push!(top, Op(pool; op=mean, window=typemax(Int)))
        push!(top, reshape2d)
        for o in 2:length(output)
            push!(top, Linear(α(output[o-1]), α(output[o]); binit=zeros, activation))
        end
        push!(top, Linear(α(output[end]), classes; binit=zeros, dropout))
    end
    push!(s, top)
    return s
end    


# TODO: Find a better way to represent types of preprocessing
torch_mobilenet_preprocess(resolution) = Op(imagenet_preprocess; normalization="torch", format="whcn", resolution)
keras_mobilenet_preprocess(resolution) = Op(imagenet_preprocess; normalization="tf", format="whcn", resolution)


function mobilenet_conv_bn(w,h,x,y; groups=1, stride=1, activation=nothing)
    c = _mobilenet_config
    p = (w-1)÷2
    padding = (w > 1 && stride > 1 && c.tfpadding ? ((p-1,p),(p-1,p)) : p)
    normalization=BatchNorm(; update=c.bnupdate, epsilon=c.bnepsilon)
    Conv(w,h,x,y; normalization, groups, stride, padding, activation)
end


function relu6(x)
    relu(x; max_value=6)
end


function mobilenet_v1_block(x, y; layout, repeat, width)
    stride = (repeat == 1 ? layout.stride : 1)
    activation = relu6
    x, y = round(Int,width*x), round(Int,width*y)
    Block(
        mobilenet_conv_bn(3, 3, 1, x; groups=x, stride, activation),
        mobilenet_conv_bn(1, 1, x, y; activation),
    )
end


const mobilenet_v1_layout = (
    (repeat=1, output=64, stride=1),
    (repeat=2, output=128, stride=2),
    (repeat=2, output=256, stride=2),
    (repeat=6, output=512, stride=2),
    (repeat=2, output=1024, stride=2),
)


function mobilenet_v2_block(x, y; layout, repeat, width)
    stride = (repeat == 1 ? layout.stride : 1)
    activation = relu6
    x, y = round(Int,width*x), round(Int,width*y)
    b = layout.expansion * x
    s = Block()
    b != x && push!(s, mobilenet_conv_bn(1, 1, x, b; activation))
    push!(s, mobilenet_conv_bn(3, 3, 1, b; groups=b, stride, activation))
    push!(s, mobilenet_conv_bn(1, 1, b, y))
    x == y ? Add(s, identity) : s
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


function mobilenet_v3_block(input, output; layout, repeat, width)
    stride = (repeat == 1 ? layout.stride : 1)
    kernel, activation = layout.kernel, layout.activation
    input, output, expand, squeeze = (x->round(Int,width*x)).((input, output, layout.expand, layout.squeeze))
    s = Block()
    channels = input
    if expand > 0
        push!(s, mobilenet_conv_bn(1, 1, channels, expand; activation))
        channels = expand
    end
    push!(s, mobilenet_conv_bn(kernel, kernel, 1, channels; activation, groups=channels, stride))
    if squeeze > 0
        push!(s, Mul(Block(                     
            Op(pool; op=mean, window=typemax(Int)),
            Conv(1, 1, channels, squeeze; binit=zeros, activation=relu),
            Conv(1, 1, squeeze, channels; binit=zeros, activation=hardsigmoid),
        ), identity))
    end
    push!(s, mobilenet_conv_bn(1, 1, channels, output))
    return input == output && stride == 1 ? Add(s, identity) : s
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

mobilenet_models = Dict{String,NamedTuple}(
    "mobilenet_v1_100_224_tf" => (width=1, resolution=224, input=32, output=(), classes=1000, activation=relu6, tfpadding=true, bnupdate=0.01, bnepsilon=0.001, dropout=0.001, block=mobilenet_v1_block, layout=mobilenet_v1_layout, preprocess=keras_mobilenet_preprocess),
    "mobilenet_v2_100_224_tf" => (width=1, resolution=224, input=32, output=1280, classes=1000, activation=relu6, tfpadding=true, bnupdate=0.001, bnepsilon=0.001, dropout=0.2, block=mobilenet_v2_block, layout=mobilenet_v2_layout, preprocess=keras_mobilenet_preprocess),
    "mobilenet_v2_100_224_pt" => (width=1, resolution=224, input=32, output=1280, classes=1000, activation=relu6, tfpadding=false, bnupdate=0.1, bnepsilon=1e-5, dropout=0.2, block=mobilenet_v2_block, layout=mobilenet_v2_layout, preprocess=torch_mobilenet_preprocess),
    "mobilenet_v3_large_100_224_pt" => (width=1, resolution=224, input=16, output=(960,1280), classes=1000, activation=hardswish, tfpadding=false, bnupdate=0.01, bnepsilon=0.001, dropout=0.2, block=mobilenet_v3_block, layout=mobilenet_v3_large_layout, preprocess=torch_mobilenet_preprocess),
    "mobilenet_v3_small_100_224_pt" => (width=1, resolution=224, input=16, output=(576,1024), classes=1000, activation=hardswish, tfpadding=false, bnupdate=0.01, bnepsilon=0.001, dropout=0.2, block=mobilenet_v3_block, layout=mobilenet_v3_small_layout, preprocess=torch_mobilenet_preprocess),
)

function MobileNet(s::String; pretrained=true)
    @assert haskey(mobilenet_models, s)  "Please choose from known MobileNet models:\n$(collect(keys(mobilenet_models)))"
    kwargs = mobilenet_models[s]
    model = MobileNet(; kwargs...)
    res = kwargs.resolution
    model(Knet.atype(zeros(Float32,res,res,3,1)))
    pretrained && setweights!(model, joinpath(@artifact_str(s), "$s.jld2"))
    return model
end
