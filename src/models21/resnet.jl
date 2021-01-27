using Knet.Layers21, Knet.Ops21
using Knet.Ops20: pool # TODO: add to ops21


struct ResNetBlock
    layers
    residual
    activation
    function ResNetBlock(ls...; residual=nothing, activation=nothing)
        new(ls, residual, activation)
    end
end


function (r::ResNetBlock)(x)
    y = x
    for l in r.layers
        y = l(y)
    end
    if r.residual !== nothing
        y = y + r.residual(x)
    end
    if r.activation !== nothing
        y = r.activation.(y)
    end
    return y
end


function ConvBN(x...; activation=nothing, o...)
    ResNetBlock(
        Conv(x...; o...),
        BatchNorm();
        activation
    )
end


function ResNetInput()
    ResNetBlock(
        ConvBN(7, 7, 3, 64; stride=2, padding=3, activation=relu),
        x->pool(x; window=3, stride=2, padding=1)
    )
end


function ResNetOutput(xchannels, classes)
    ResNetBlock(
        x->pool(x; mode=1, window=(size(x,1),size(x,2))),
        Dense(classes, xchannels)
    )
end


function ResNetBasicBlock(xchannels, ychannels; stride = 1, residual = identity, activation = relu)
    ResNetBlock(
        ConvBN(3, 3, xchannels, ychannels; stride, activation=relu),
        ConvBN(3, 3, ychannels, ychannels);
        residual,
        activation
    )
end


function ResNetBasic(nlayers...; classes = 1000)
    layers = []
    push!(layers, ResNetInput())
    for (layer, nblocks) in enumerate(nlayers)
        for block in 1:nblocks
            ychannels = 2^(5+layer)      # 64, 128, 256, 512
            if layer > 1 && block == 1
                xchannels = ychannels ÷ 2
                stride = 2
                residual = ConvBN(1, 1, xchannels, ychannels; stride)
            else
                xchannels = ychannels
                stride = 1
                residual = identity
            end
            push!(layers, ResNetBasicBlock(xchannels, ychannels; stride, residual))
        end
    end
    push!(layers, ResNetOutput(2^(5+length(nlayers)), classes))
    ResNetBlock(layers...)
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
    ResNetBlock(
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
    return ResNetBlock(layers...)
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
