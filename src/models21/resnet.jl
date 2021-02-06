import Knet, AutoGrad, Images, FileIO
using Knet.Layers21: Conv, BatchNorm, Linear, Sequential, Residual
using Knet.Ops20: pool # TODO: add pool to ops21
using Knet.Ops21: relu # TODO: define activation layer?


ConvBN(x...; o...) = Conv(x...; o..., normalization=BatchNorm())


function ResNetInput()
    Sequential(
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


function ResNetBasicBlock(xchannels, ychannels; activation=relu, padding=1)
    stride = (xchannels === ychannels ? 1 : 2)
    f1 = Sequential(
        ConvBN(3, 3, xchannels, ychannels; stride, padding, activation),
        ConvBN(3, 3, ychannels, ychannels; padding),
    )
    f2 = (xchannels === ychannels ? identity :
          ConvBN(1, 1, xchannels, ychannels; stride))
    return Residual(f1, f2; activation)
end


function ResNetBasic(nblocks...; classes = 1000, channels = 64)
    s = Sequential(ResNetInput(); name="ResNetBasic$nblocks")
    for (layer, nb) in enumerate(nblocks)
        channels = layer > 1 ? 2*channels : channels # 64, 128, 256, 512
        blocks = Sequential(; name="Blocks$channels")
        for block in 1:nb
            xchannels = (layer > 1 && block == 1 ? channels ÷ 2 : channels)
            push!(blocks, ResNetBasicBlock(xchannels, channels))
        end
        push!(s, blocks)
    end
    push!(s, ResNetOutput(channels, classes))
end


function ResNetBottleneckBlock(xchannels, ychannels; activation=relu, padding=1)
    stride = (xchannels === ychannels ? 1 : 2)
    f1 = Sequential(
        ConvBN(3, 3, xchannels, ychannels; stride, padding, activation),
        ConvBN(3, 3, ychannels, ychannels; padding),
    )
    f2 = (xchannels === ychannels ? identity :
          ConvBN(1, 1, xchannels, ychannels; stride))
    return Residual(f1, f2; activation)
end


ResNetPreprocess(file::String)=ResNetPreprocess(load(file))

function ResNetPreprocess(img::Matrix{<:RGB})
    img = imresize(img, ratio=256/minimum(size(img))) # min(h,w)=256
    hcenter,vcenter = size(img) .>> 1
    img = img[hcenter-111:hcenter+112, wcenter-111:wcenter+112] # h,w=224,224
    img = channelview(img)                                      # c,h,w=3,224,224
    μ,σ = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img = (img .- μ) ./ σ
    img = permutedims(img, (3,2,1)) # 224,224,3
    img = reshape(img, (size(img)..., 1)) # 224,224,3,1
    Knet.atype(img)
end

#=

function ResNetBottleneckBlock(
    xchannels, ychannels;
    stride = 1,
    groups = 1,
    dilation = 1,
    base_width = 64,            # ???
    residual = identity,
    activation = relu,
)
    expansion = 4               # ???
    width = groups * ychannels * base_width ÷ 64 # ???
    Sequential(
        ConvBN(1, 1, xchannels, width; activation=relu), 
        ConvBN(3, 3, width, width; stride, groups, dilation, activation=relu), 
        ConvBN(1, 1, xchannels, ychannels * expansion);
        residual,
        activation,
    )
end


function ResNetLayer(xchannels, ychannels, blocks, block)
    layers = []
    push!(layers, block(xchannels, ychannels))
    for i in 1:blocks
        push!(layers, block(ychannels, ychannels))
    end
    return Sequential(layers...)
end


function ResNet(
    blocks...;
    classes = 1000,
)
    layers = []
    
end


    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))



"""

* Code https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
* ResNet https://arxiv.org/abs/1512.03385
* ResNext https://arxiv.org/abs/1611.05431
"""
struct ResNet
end

@enum ResNetModels begin
    resnet18
    resnet34
    resnet50
    resnet101
    resnet152
    resnet50v2
    resnet101v2
    resnet152v2
    resnext50
    resnext101
end

struct ResNetBasicBlock
    conv1
    bn1
    conv2
    bn2
    downsample
end

struct ResNetBottleneckBlock

end


input layer:
conv1: 3=>64, kernel=7, stride=2, pad=3, bias=false
maxpool: kernel=3, stride=2, pad=1

layer_i.block_1:
inplanes = self.inplanes
planes = planes
** stride(1) = stride
** downsample(None) = downsample
groups(1) = self.groups
base_width(64) = self.base_width
** dilation(1) = previous_dilation
norm_layer(None) = norm_layer

layer_i.block_j:
inplanes = self.inplanes
planes = planes
** stride(1) = 
** downsample(None) = 
groups(1) = self.groups
base_width(64) = self.base_width
** dilation(1) = self.dilation
norm_layer(None) = norm_layer


self._norm_layer
self.dilation
stride=1
conditional downsample based on self.inplanes != planes * block.expansion

self.groups
self.base_width
=#
