export EfficientNet

import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Block, Op, Add, Mul
using Knet.Ops21: swish, sigm, pool, mean, reshape2d
using Artifacts

# TODO: drop_connect_rate * b / blocks

"""
    EfficientNet(; kwargs...)
    EfficientNet(name::String; pretrained=true)

Pretrained models:

    name            top1   ref1   size
    ----            -----  -----  ----
    efficientnetb0  77.04  77.1    21M
    efficientnetb1  79.03  79.1    31M
    efficientnetb2  79.97  80.1    36M
    efficientnetb3  81.30  81.6    48M
    efficientnetb4  82.56  82.9    75M
    efficientnetb5  83.52  83.6   118M
    efficientnetb6  83.80  84.0   166M
    efficientnetb7  84.02  84.3   256M
                               
References:
* [Tan & Le 2019](https://arxiv.org/abs/1905.11946) EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. (ICML 2019)
* [Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet.py)
"""
function EfficientNet(
    ;
    width = 1,
    depth = 1,
    resolution = 224,
    input = 32,
    output = 1280,
    classes = 1000,
    dropout = 0.2,
    tfpadding = true, # wrong padding costs ~1%
    bnupdate = 0.01,
    bnepsilon = 0.001,
    normalize = efficientnet_normalize,
)
    α(x) = ceil(Int, depth * x)
    β(x) = (wx = width*x; qx = max(8, floor(Int, wx + 4) ÷ 8 * 8); qx < 0.9 * wx ? qx+8 : qx)
    global efficientnet_config = (; tfpadding, bnupdate, bnepsilon)
    b = Block()
    push!(b, Op(imagenet_preprocess; resolution, normalize))
    push!(b, efficientnet_conv(3, β(input); kernel=3, stride=2, resolution))
    resolution = (1+resolution) ÷ 2
    for l in efficientnet_layout
        stride = l.stride
        for r in 1:α(l.repeat)
            push!(b, efficientnet_block(β(input), β(l.output); stride, resolution, l.kernel, l.expand, l.squeeze))
            resolution = (stride == 2 ? (1+resolution) ÷ 2 : resolution)
            stride, input = 1, l.output
        end
    end
    push!(b, efficientnet_conv(β(input), β(output); kernel=1))
    push!(b, Block(
        Op(pool; op=mean, window=typemax(Int)), reshape2d,
        Linear(β(output), classes; binit=zeros, dropout)))
    return b
end


function efficientnet_block(input, output; stride, resolution, kernel, expand, squeeze)
    b = Block()
    c = input * expand
    if expand != 1
        push!(b, efficientnet_conv(input, c; kernel=1))
    end
    push!(b, efficientnet_conv(1, c; groups=c, stride, kernel, resolution))
    if squeeze != 1
        push!(b, efficientnet_se(c, max(1, input÷squeeze)))
    end
    push!(b, efficientnet_conv(c, output; kernel=1, activation=nothing))
    return (input == output && stride == 1 ? Add(b, identity) : b)
end


function efficientnet_conv(input, output; kernel=1, groups=1, stride=1, activation=swish, resolution=0)
    c = efficientnet_config
    p,q = (kernel-1)÷2, 1-resolution%2
    padding = (c.tfpadding && q > 0 && kernel > 1 && stride > 1 ? ((p-q,p),(p-q,p)) : p)
    normalization=BatchNorm(; update=c.bnupdate, epsilon=c.bnepsilon)
    Conv(kernel,kernel,input,output; normalization, groups, stride, padding, activation)
end


function efficientnet_se(input, squeeze; activation=swish)
    Mul(Block(                     
        Op(pool; op=mean, window=typemax(Int)),
        Conv(1, 1, input, squeeze; binit=zeros, activation),
        Conv(1, 1, squeeze, input; binit=zeros, activation=sigm),
    ), identity)
end


# Use this to replicate the currently buggy implementation in tf.keras.applications
efficientnet_normalize_buggy(x) = (x .- [0.485, 0.456, 0.406]) ./ sqrt.([0.229, 0.224, 0.225])

# Use this to get better results
efficientnet_normalize(x) = (x .- [0.485, 0.456, 0.406]) ./ [0.229, 0.224, 0.225]


efficientnet_layout = (
    (repeat=1, input=32,  output=16,  kernel=3, stride=1, expand=1, squeeze=4), # 112x112
    (repeat=2, input=16,  output=24,  kernel=3, stride=2, expand=6, squeeze=4), # 112x112
    (repeat=2, input=24,  output=40,  kernel=5, stride=2, expand=6, squeeze=4), # 56x56
    (repeat=3, input=40,  output=80,  kernel=3, stride=2, expand=6, squeeze=4), # 28x28
    (repeat=3, input=80,  output=112, kernel=5, stride=1, expand=6, squeeze=4), # 14x14
    (repeat=4, input=112, output=192, kernel=5, stride=2, expand=6, squeeze=4), # 7x7
    (repeat=1, input=192, output=320, kernel=3, stride=1, expand=6, squeeze=4), # 7x7
)


efficientnet_models = Dict{String,NamedTuple}(
    "efficientnetb0" => (width=1.0, depth=1.0, resolution=224, dropout=0.2),
    "efficientnetb1" => (width=1.0, depth=1.1, resolution=240, dropout=0.2),
    "efficientnetb2" => (width=1.1, depth=1.2, resolution=260, dropout=0.3),
    "efficientnetb3" => (width=1.2, depth=1.4, resolution=300, dropout=0.3),
    "efficientnetb4" => (width=1.4, depth=1.8, resolution=380, dropout=0.4),
    "efficientnetb5" => (width=1.6, depth=2.2, resolution=456, dropout=0.4),
    "efficientnetb6" => (width=1.8, depth=2.6, resolution=528, dropout=0.5),
    "efficientnetb7" => (width=2.0, depth=3.1, resolution=600, dropout=0.5),
)


function EfficientNet(s::String; pretrained=true)
    @assert haskey(efficientnet_models, s)  "Please choose from known EfficientNet models:\n$(collect(keys(efficientnet_models)))"
    kwargs = efficientnet_models[s]
    model = EfficientNet(; kwargs...)
    res = kwargs.resolution
    if pretrained
        model(Knet.atype(zeros(Float32,res,res,3,1)))
        setweights!(model, joinpath(@artifact_str(s), "$s.jld2"))
    end
    return model
end


