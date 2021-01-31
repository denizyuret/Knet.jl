import Knet.Layers21: Conv, BatchNorm
using AutoGrad: value

using PyCall
include("resnet.jl")
torch = pyimport("torch")
models = pyimport("torchvision.models")
a18 = models.resnet18(pretrained=true)


function Conv(p::PyObject)
    Conv(cweight(p.weight); bias = bweight(p.bias),
         p.padding, p.stride, p.dilation, p.groups)
end

function BatchNorm(p::PyObject)
    Batchnorm(
        ; epsilon = b.eps,
        momentum = 1 - b.momentum,
        mean = value(nweight(b.running_mean)),
        var = value(nweight(b.running_var)),
        bias = nweight(b.bias),
        scale = nweight(b.weight),
    )
end

function ConvBN(pc::PyObject, pb::PyObject; activation = nothing)
    Sequential(
        Conv(pc),
        BatchNorm(pb),
        activation(pa)
    )
end


# 1. write a torchimport.jl with Conv, Dense, BatchNorm, ResNetInput etc. constructors from PyObjects
#    pros: more robust
#    cons: model structure may resemble pytorch
# 2. try to simply transfer weights, need correspondence of layers, non-parameters like bn.mean
#    pros: flexibility in model definition
#    cons: each time we change model, import code will be invalid.

# One can get the list of parameters in torch with:
# for name, param in m.named_parameters(): # or m.parameters()
#   print(name, param)

# a18 = models.resnet18(pretrained=true)
# b18 = ResNetBasic(2,2,2,2)
# bx = rand(Float32, 224, 224, 3, 1)
# by = b18(bx)


# rx = rand(Float32, 1, 3, 224, 224)
# tx = torch.tensor(rx)

# ty = t18(tx)
# @show(ty.shape, ty.dtype, ty.device)
# ry = ty.detach().numpy()

# kx = permutedims(rx, (4,3,2,1))
# ky = r18(kx)


# xs = pyimport("torchvision.transforms")
# xform = xs.Compose(
#     [ xs.Resize(256),
#       xs.CenterCrop(224),
#       xs.ToTensor(),
#       xs.Normalize(mean=[0.485, 0.456, 0.406],
#                    std =[0.229, 0.224, 0.225]),
#       ]
# )

# img = load("/datasets/ImageNet/ILSVRC2015/Data/CLS-LOC/train/n01440764/n01440764_10026.JPEG")

nothing
