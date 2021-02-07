export ResNet
import Knet
using Knet.Layers21: Conv, BatchNorm, Linear, Sequential, Residual
using Knet.Ops20: pool, softmax # TODO: add pool to ops21
using Knet.Ops21: relu # TODO: define activation layer?
using Artifacts


"""
    ResNet(; nblocks=(2,2,2,2), block=ResNetBottleneck, groups=1, bottleneck=1, classes=1000)
    ResNet(name::String; pretrained=true)

Return a ResNet model based on keyword arguments or a name specifying a predefined
structure. Load pretrained weights if `pretrained=true`.

A ResNet model consists of an input block, four layers, and an output block. The input and
output blocks are the same for every model:

    Input:
    resnetinput
    Conv(7×7, 3=>64, padding=3, stride=2, BatchNorm(), relu)
    x->pool(x; window=3, stride=2, padding=1)

    Output:
    x->pool(x; mode=1, window=size(x)[1:2])
    x->reshape(x, :, size(x,4))
    Linear(classes, bias)
    
The `resnetinput` function performs preprocessing on the input, the user can define their own
methods to handle different types of input (images, files, etc.) with resizing,
normalization etc.

Of the four layers, layer `i` contains `nblocks[i]` blocks of type `block` which can be
`ResNetBottleneck` (default) or `ResNetBasic`. `ResNetBasic` is a residual block with two
3×3 convolutions. `ResNetBottleneck` is a residual block with a 1×1 convolution that brings
the number of channels down, a 3×3 convolution (possibly grouped if `groups > 1`), and
another 1×1 convolution that brings the number of channels up. The keyword argument
`bottleneck` controls the ratio of the output channels to the intermediate channels. The
blocks in layer `i` have `2^(i+5)` output channels in `ResNetBasic` and `2^(i+7)` output
channels in `ResNetBottleneck`.  The 3×3 convolution of each layer except the first has
`stride=2` which brings the spatial resolution down.

The pretrained models come from
[torchvision](https://pytorch.org/vision/stable/models.html).  For pretrained models, the
expected size of the input is (224,224,3,N) with pixel values should be normalized using:

    (rgb_pixel_0_1 .- [0.485, 0.456, 0.406]) ./ [0.229, 0.224, 0.225]

Other sizes will work but may not give good classification accuracy. The number of classes
is 1000. The class labels, as well as training, validation and test sets can be found at
[Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge). Here are all the
predefined models with name, settings, size in bytes, runtime compared to resnet18, top-1
validation error.

    name              settings                                      size  time  top1
    ----              --------                                      ----  ----  ----
    resnet18          (nblocks=(2,2,2,2), block=ResNetBasic)         45M  1.00  .3106
    resnet34          (nblocks=(3,4,6,3), block=ResNetBasic)         84M  1.38  .2724
    resnet50          (nblocks=(3,4,6,3), bottleneck=4)              98M  2.44  .2472
    resnet101         (nblocks=(3,4,23,3), bottleneck=4)            171M  3.85  .2317
    resnet152         (nblocks=(3,8,36,3), bottleneck=4)            231M  5.35  .2220
    wide_resnet50_2   (nblocks=(3,4,6,3), bottleneck=2)             264M  5.14  .2225
    wide_resnet101_2  (nblocks=(3,4,23,3), bottleneck=2)            485M  7.33  .2181
    resnext50_32x4d   (nblocks=(3,4,6,3), groups=32, bottleneck=2)   96M  3.26  .2294
    resnext101_32x8d  (nblocks=(3,4,23,3), groups=32)               340M  8.41  .2131

Note: The errors are slightly different from the ones given by torchvision, which is
probably due to the differences in preprocessing, in particular `imresize` in Julia gives
slightly different results compared to `torchvision.transforms`.

References:
* He, Kaiming et al. "Deep Residual Learning for Image Recognition." 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 770-778.
* Zagoruyko, Sergey and Nikos Komodakis. "Wide Residual Networks." ArXiv abs/1605.07146 (2016)
* Xie, Saining et al. "Aggregated Residual Transformations for Deep Neural Networks." 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 5987-5995.

"""
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
        resnetinput,
        ConvBN(7, 7, 3, 64; stride=2, padding=3, activation=relu),
        x->pool(x; window=3, stride=2, padding=1);
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


# Preprocessing - override this to handle image, file, url etc. as input
resnetinput(x) = Knet.atype(x)


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
