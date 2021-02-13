export ResNet
import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Sequential, Residual
using Knet.Ops20: pool # TODO: add pool to ops21
using Knet.Ops21: relu # TODO: define activation layer?
using Artifacts


"""
    ResNet(; nblocks=(2,2,2,2), block=ResNetBottleneck, groups=1, bottleneck=1, classes=1000)
    ResNet(name::String; pretrained=true)

Return a ResNet model based on keyword arguments or a name specifying a predefined
structure. Load pretrained weights if `pretrained=true`, randomly initialize otherwise.

A ResNet model `r` applied to an input of size `(width,height,channels=3,images)` returns a
`(classes,1)` vector of class scores. The model is a sequence of 6 functions: `r[1]` is the
input block, `r[2:5]` are four stages of residual blocks, and `r[6]` is the output
block. One can manipulate a ResNet model using the array interface, e.g. `r[1:5](img)` would
give the output of the last residual block, `pushfirst!(r, preprocess)` would add the
`preprocess` function to the beginning, etc.

The input and output blocks are the same for every ResNet model:

    Input:
        Conv(7×7, 3=>64, padding=3, stride=2, BatchNorm(), relu)
        x->pool(x; window=3, stride=2, padding=1)

    Output:
        x->pool(x; mode=1, window=size(x)[1:2])
        x->reshape(x, :, size(x,4))
        Linear(classes, bias)
    
Stage i in 1:4 consists of `nblocks[i]` blocks of type `block` which can be
`ResNetBottleneck` (default) or `ResNetBasic`. `ResNetBasic` is a residual block with two
3×3 convolutions. `ResNetBottleneck` is a residual block with a 1×1 convolution that brings
the number of channels down, a 3×3 convolution (possibly grouped if `groups > 1`), and
another 1×1 convolution that brings the number of channels up. The keyword argument
`bottleneck` controls the ratio of the output channels to the intermediate channels. The
number of channels is doubled at each stage: blocks in the i'th stage have `2^(i+5)` output
channels for `ResNetBasic` and `2^(i+7)` output channels for `ResNetBottleneck`.  The
spatial resolution is halved at each stage: the first 3×3 convolution at each stage except
the first has `stride=2`.

The pretrained models come from
[torchvision](https://pytorch.org/vision/stable/models.html). A generic ResNet model can
handle any size input and any number of classes, but for pretrained models the training
settings should be replicated for best results: The number of classes is 1000 and the
expected size of the input is (224,224,3,N). The input images are resized to 256 pixels at
the short edge and center cropped to size (224,224). The pixel values are transformed from
the (0,1) range to:

    (rgb_pixel_0_1 .- [0.485, 0.456, 0.406]) ./ [0.229, 0.224, 0.225]

The class labels, as well as training, validation and test sets can be found at
[Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge). Here are all the
pretrained models with name, settings, size in bytes, flops compared to resnet18, and top-1
validation error.

    name              settings                                      size  flops  top1
    ----              --------                                      ----  -----  ----
    resnet18          (nblocks=(2,2,2,2), block=ResNetBasic)         45M   1.00  .3106
    resnet34          (nblocks=(3,4,6,3), block=ResNetBasic)         84M   1.38  .2724
    resnet50          (nblocks=(3,4,6,3), bottleneck=4)              98M   2.44  .2472
    resnet101         (nblocks=(3,4,23,3), bottleneck=4)            171M   3.85  .2317
    resnet152         (nblocks=(3,8,36,3), bottleneck=4)            231M   5.35  .2220
    wide_resnet50_2   (nblocks=(3,4,6,3), bottleneck=2)             264M   5.14  .2225
    wide_resnet101_2  (nblocks=(3,4,23,3), bottleneck=2)            485M   7.33  .2181
    resnext50_32x4d   (nblocks=(3,4,6,3), groups=32, bottleneck=2)   96M   3.26  .2294
    resnext101_32x8d  (nblocks=(3,4,23,3), groups=32)               340M   8.41  .2131

Note: The errors are slightly different from the ones given by torchvision, which is
probably due to the differences in preprocessing, in particular `imresize` in Julia gives
different results compared to `Resize` in `torchvision.transforms`.

References:
* He, Kaiming et al. "Deep Residual Learning for Image Recognition." 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 770-778.
* Zagoruyko, Sergey and Nikos Komodakis. "Wide Residual Networks." ArXiv abs/1605.07146 (2016)
* Xie, Saining et al. "Aggregated Residual Transformations for Deep Neural Networks." 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 5987-5995.

"""
function ResNet(; nblocks = (2,2,2,2), block = ResNetBottleneck, groups = 1, bottleneck = 1, classes = 1000)
    s = Sequential(ResNetInput(); name="$block$nblocks")
    x, y = 64, (block === ResNetBasic ? 64 : 256)
    for (stage, nblock) in enumerate(nblocks)
        if stage > 1; y *= 2; end
        b = y ÷ bottleneck
        blocks = Sequential(; name="Stage$stage")
        for iblock in 1:nblock
            stride = (stage > 1 && iblock == 1) ? 2 : 1
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
        ConvBN(7, 7, 3, 64; stride=2, padding=3, activation=relu),
        Op(pool; window=3, stride=2, padding=1);
        name = "Input"
    )
end


function ResNetOutput(xchannels, classes)
    Sequential(
        x->pool(x; mode=1, window=size(x)[1:2]),
        x->reshape(x, :, size(x,4)),
        Linear(xchannels, classes; binit=zeros); # TODO: rethink how to specify bias in Linear/Conv
        name = "Output"
    )
end


# Frequently used combo
ConvBN(x...; o...) = Conv(x...; o..., normalization=BatchNorm())


# Run a single image so weights get initialized
resnetinit(m) = (m(Knet.atype(zeros(Float32,224,224,3,1))); m)


# Pretrained models from torchvision
resnetmodels = Dict{String,NamedTuple}(
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
