#module MobileNetModule
export MobileNet

import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Sequential, Op
using Knet.Ops20: pool
using Knet.Ops21: relu, dropout
using HDF5


"""
References:
* https://arxiv.org/abs/1704.04861
* https://arxiv.org/abs/1801.04381
* https://arxiv.org/abs/1905.02244
"""
function MobileNet(; width = 1, resolution = 224)
    α(x) = round(Int, width*x)
    m = Sequential(
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
        Op(dropout, 0.001),
        Linear(α(1024), 1000; binit=zeros)
    )
end


function DWConv(x, y; stride = 1)
    Sequential(
        ConvBN6(3, 3, 1, x; groups=x, stride),
        ConvBN6(1, 1, x, y),
    )
end


function ConvBN6(w,h,x,y; stride = 1, o...)
    padding = (w == 1 ? 0 : stride == 1 ? 1 : ((0,1),(0,1)))
    Conv(w,h,x,y; stride, padding,
         normalization=BatchNorm(; update=0.01, epsilon=0.001),
         activation=Op(relu; max_value=6), o...)
end


# mobilenet models from keras.applications
mobilenetmodels = Dict{String,NamedTuple}(
    "mobilenet_100_224" => (width=1, resolution=224),
)


function MobileNet(s::String; pretrained=true)
    @assert haskey(mobilenetmodels, s)  "Unknown MobileNet model $s"
    kwargs = mobilenetmodels[s]
    model = MobileNet(; kwargs...)
    model(Knet.atype(zeros(Float32,224,224,3,1)))
    pretrained && setweights!(model, joinpath(@artifact_str(s), "$s.jld2"))
    return model
end

#end # MobileNetModule
