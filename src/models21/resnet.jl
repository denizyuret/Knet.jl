export ResNet
import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Sequential, Residual
using Knet.Ops20: pool, softmax # TODO: add pool to ops21
using Knet.Ops21: relu # TODO: define activation layer?
using Artifacts


function ResNet(; nblocks = (2,2,2,2), block = ResNetBottleneck, groups = 1, bottleneck = 1, classes = 1000)
    s = Sequential(ResNetInput(); name="$block$nblocks")
    x, y = 64, (block === ResNetBasic ? 64 : 256)
    for (layer, nblock) in enumerate(nblocks)
        if layer > 1; y *= 2; end
        b = y ÷ bottleneck
        blocks = Sequential(; name="Layer$layer")
        for iblock in 1:nblock
            stride = (layer > 1 && iblock == 1) ? 2 : 1
            push!(blocks, block(x, b, y; stride, groups))
            x = y
        end
        push!(s, blocks)
    end
    push!(s, ResNetOutput(y, classes))
    resnetinit(s)
end


function ResNetBottleneck(x, b, y; activation=relu, groups=1, padding=1, stride=1, o...)
    Residual(
        Sequential(
            ConvBN(1, 1, x, b; activation),
            ConvBN(3, 3, b÷groups, b; activation, groups, padding, stride),
            ConvBN(1, 1, b, y),
        ),
        (x != y ? ConvBN(1, 1, x, y; stride) : identity);
        activation)
end


function ResNetBasic(x, b, y; stride=1, padding=1, activation=relu, o...)
    Residual(
        Sequential(
            ConvBN(3, 3, x, b; activation, padding, stride),
            ConvBN(3, 3, b, y; padding),
        ),
        (x != y ? ConvBN(1, 1, x, y; stride) : identity);
        activation)
end


function ResNetInput()
    Sequential(
        resnetprep,
        ConvBN(7, 7, 3, 64; stride=2, padding=3, activation=relu),
        x->pool(x; window=3, stride=2, padding=1);
        name = "Input"
    )
end


function ResNetOutput(xchannels, classes)
    Sequential(
        x->pool(x; mode=1, window=(size(x,1),size(x,2))),
        x->reshape(x, :, size(x,4)),
        Linear(xchannels, classes; binit=zeros); # TODO: rethink how to specify bias in Linear/Conv
        name = "Output"
    )
end


# Frequently used combo
ConvBN(x...; o...) = Conv(x...; o..., normalization=BatchNorm())


# Run a single image so weights get initialized
resnetinit(m) = (m(Knet.atype(zeros(Float32,224,224,3,1))); m)


# Preprocessing - override this to handle image, file, url etc. as input
resnetprep(x) = Knet.atype(x)


# Pretrained models from torchvision
const resnetmodels = Dict{String,NamedTuple}(
    "resnet18" => (nblocks=(2,2,2,2), block=ResNetBasic),
    "resnet34" => (nblocks=(3,4,6,3), block=ResNetBasic),
    "resnet50" => (nblocks=(3,4,6,3), bottleneck=4),
    "resnet101" => (nblocks=(3,4,23,3), bottleneck=4),
    "resnet152" => (nblocks=(3,8,36,3), bottleneck=4),
    "wide_resnet50_2" => (nblocks=(3,4,6,3), bottleneck=2),
    "wide_resnet101_2" => (nblocks=(3,4,23,3), bottleneck=2),
    "resnext50_32x4d" => (nblocks=(3,4,6,3), groups=32, bottleneck=2),
    "resnext101_32x8d" => (nblocks=(3,4,23,3), groups=32),
)

function ResNet(s::String; pretrained=true)
    @assert haskey(resnetmodels, s)  "Unknown ResNet model $s"
    kwargs = resnetmodels[s]
    model = ResNet(; kwargs...)
    pretrained && setweights!(model, joinpath(@artifact_str(s), "$s.jld2"))
    return model
end
