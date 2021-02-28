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
    preprocess = torch_mobilenet_preprocess, # keras_mobilenet_preprocess for keras models
    padding = 1,                # ((0,1),(0,1)) for keras models
    bnupdate = 0.1,             # torch:0.1, keras.MobileNetV1:0.01, keras.MobileNetV2:0.001
    bnepsilon = 1e-5,           # torch:1e-5, keras:0.001
)
    α(x) = round(Int, width*x)
    s = Sequential(MobileNetInput(preprocess(resolution), α(input); padding, bnupdate, bnepsilon))
    channels = input
    for (repeat, outchannels, stride, expansion) in layout
        for r in 1:repeat
            push!(s, block(α(channels), α(outchannels); stride, expansion, padding, bnupdate, bnepsilon))
            channels = outchannels
            stride = 1
        end
    end
    channels != output && push!(s, ConvBN6(1, 1, α(channels),  α(output); padding, bnupdate, bnepsilon))
    push!(s, MobileNetOutput(α(output), classes))
    return s
end    


torch_mobilenet_preprocess(resolution) = Op(imagenet_preprocess; normalization="torch", format="whcn", resolution)
keras_mobilenet_preprocess(resolution) = Op(imagenet_preprocess; normalization="tf", format="whcn", resolution)


function MobileNetInput(preprocess, input; o...)
    Sequential(
        preprocess,
        ConvBN6(3, 3, 3, input; stride = 2, o...)
    )
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


function MobileNetOutput(output, classes)
    Sequential(
        x->pool(x; mode=1, window=size(x)[1:2]),
        Op(reshape, (output,:)),
        Linear(output, classes; binit=zeros)
    )
end


function ConvBN6(w,h,x,y; groups = 1, stride = 1, padding = 1, bnupdate=0.1, bnepsilon=1e-5,
                 activation=Op(relu; max_value=6))
    padding = (w == 1 ? 0 : stride == 1 ? 1 : padding)
    normalization=BatchNorm(; update=bnupdate, epsilon=bnepsilon)
    Conv(w,h,x,y; normalization, groups, stride, padding, activation)
end


function MobileNet(s::String; pretrained=true)
    @assert haskey(mobilenetmodels, s)  "Unknown MobileNet model $s"
    kwargs = mobilenetmodels[s]
    model = MobileNet(; kwargs...)
    model(Knet.atype(zeros(Float32,224,224,3,1)))
    pretrained && setweights!(model, "$s.jld2") # joinpath(@artifact_str(s), "$s.jld2")) ##DBG
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


### DEPRECATED:

function MobileNetV1(; width = 1, resolution = 224)
    α(x) = round(Int, width*x)
    Sequential(
        Op(imagenet_preprocess; normalization="tf", format="whcn", resolution),
        ConvBN6(3, 3, 3, α(32); stride = 2),

        DWConv(α(32),  α(64)),

        DWConv(α(64),  α(128);  stride = 2),
        DWConv(α(128), α(128)),

        DWConv(α(128), α(256);  stride = 2),
        DWConv(α(256), α(256)),

        DWConv(α(256), α(512);  stride = 2),
        DWConv(α(512), α(512)),
        DWConv(α(512), α(512)),
        DWConv(α(512), α(512)),
        DWConv(α(512), α(512)),
        DWConv(α(512), α(512)),

        DWConv(α(512), α(1024); stride = 2),
        DWConv(α(1024), α(1024)),

        x->pool(x; mode=1, window=size(x)[1:2]),
        Op(reshape, (α(1024),:)),
        Linear(α(1024), 1000; binit=zeros)
    )
end

function MobileNetV2(; width = 1, resolution = 224)
    α(x) = round(Int, width*x)
    Sequential(
        Op(imagenet_preprocess; normalization="tf", format="whcn", resolution),
        ConvBN6(3, 3, 3, α(32); stride = 2),  # 224=>112

        MBConv(α(32), α(16); expansion = 1),

        MBConv(α(16),  α(24); stride = 2),    # 112=>56
        MBConv(α(24),  α(24)),

        MBConv(α(24),  α(32); stride = 2),    # 56=>28
        MBConv(α(32),  α(32)),
        MBConv(α(32),  α(32)),

        MBConv(α(32),  α(64); stride = 2),    # 28=>14
        MBConv(α(64),  α(64)),
        MBConv(α(64),  α(64)),
        MBConv(α(64),  α(64)),
 
        MBConv(α(64),  α(96)),                # 14=>14
        MBConv(α(96),  α(96)),
        MBConv(α(96),  α(96)),

        MBConv(α(96),  α(160); stride = 2),   # 14=>7
        MBConv(α(160),  α(160)),
        MBConv(α(160),  α(160)),
        
        MBConv(α(160),  α(320)),              # 7=>7

        ConvBN6(1, 1, α(320),  α(1280)),      # 7=>7
        x->pool(x; mode=1, window=size(x)[1:2]),
        Op(reshape, (α(1280),:)),
        Linear(α(1280), 1000; binit=zeros),
    )
end



#end # MobileNetModule
