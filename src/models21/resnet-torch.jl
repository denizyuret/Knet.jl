import Knet.Layers21: Conv, BatchNorm
import Knet.Train20: param
import CUDA: CuArray
import NNlib, Knet, AutoGrad
using PyCall

include("resnet.jl")
torch = pyimport("torch")
nn = pyimport("torch.nn")
models = pyimport("torchvision.models")
t2a(x) = x.cpu().detach().numpy()
Base.isapprox(a::Array,b::CuArray)=isapprox(a,Array(b))
Base.isapprox(a::AutoGrad.Value,b::AutoGrad.Value)=isapprox(a.value,b.value)
chkparams(a,b)=all(isapprox(pa,pb) for (pa,pb) in zip(AutoGrad.params(a),AutoGrad.params(b)))


function ResNetBasic(p::PyObject)
    layers = (p.layer1, p.layer2, p.layer3, p.layer4)
    s = Sequential(ResNetInput(p.conv1, p.bn1); name="ResNetBasic$(length.(layers))")
    for (i, layer) in enumerate(layers)
        channels = 2^(i+5)
        blocks = Sequential(; name="Blocks$channels")
        for block in layer
            push!(blocks, ResNetBasicBlock(block))
        end
        push!(s, blocks)
    end
    push!(s, ResNetOutput(p.fc))
end


function ResNetBasicBlock(p::PyObject)
    bn1 = BatchNorm(p.bn1)
    bn2 = BatchNorm(p.bn2)
    conv1 = Conv(p.conv1; normalization=bn1, activation=NNlib.relu)
    conv2 = Conv(p.conv2; normalization=bn2)
    f1 = Sequential(conv1, conv2)
    if p.downsample === nothing
        f2 = identity
    else
        bn3 = BatchNorm(p.downsample[2])
        f2 = Conv(p.downsample[1]; normalization=bn3)
    end
    return Residual(f1, f2; activation=NNlib.relu)
end


function ResNetInput(conv1::PyObject, bn1::PyObject)
    bn1 = BatchNorm(bn1)
    Sequential(
        Conv(conv1; normalization=bn1, activation=NNlib.relu),
        x->NNlib.maxpool(x, (3,3); stride=2, pad=1);
        name = "Input"
    )
end


function ResNetOutput(fc::PyObject)
    w = param(t2a(fc.weight))
    bias = param(t2a(fc.bias))
    Sequential(
        x->NNlib.meanpool(x, (size(x,1),size(x,2))),
        x->reshape(x, :, size(x,4)),
        Dense(w; bias);
        name = "Output"
    )
end


function Conv(p::PyObject; normalization=nothing, activation=nothing, crosscorrelation=true)
    w = param(permutedims(t2a(p.weight), (4,3,2,1)))
    bias = (p.bias === nothing ? nothing :
            param(reshape(t2a(p.bias), (1,1,:,1))))
    Conv(w; bias, normalization, activation, crosscorrelation,
         p.padding, p.stride, p.dilation, p.groups)
end


function BatchNorm(b::PyObject)
    bnweight(x) = param(reshape(t2a(x), (1,1,:,1)))
    BatchNorm(
        ; epsilon = b.eps,
        momentum = 1 - b.momentum,
        mean = bnweight(b.running_mean).value,
        var = bnweight(b.running_var).value,
        bias = bnweight(b.bias),
        scale = bnweight(b.weight),
    )
end


p18 = models.resnet18(pretrained=true).eval()
px = rand(Float32,2,3,224,224)
py = p18(torch.tensor(px)) |> t2a


Knet.atype() = Array{Float32}
a18 = ResNetBasic(p18)
ax = Knet.atype(permutedims(px, (4,3,2,1)))
ay = a18(ax)

# Knet.atype() = KnetArray{Float32}
# k18 = ResNetBasic(p18)
# kx = Knet.atype(permutedims(px, (4,3,2,1)))
# ky = k18(kx)

Knet.atype() = CuArray{Float32}
c18 = ResNetBasic(p18)
cx = Knet.atype(permutedims(px, (4,3,2,1)))

@show chkparams(a18,c18)

cx5 = c18[1:5](cx)

@show chkparams(a18,c18)

cx6 = c18[6](cx5)

@show chkparams(a18,c18)


# @show isapprox(Array(ay), Array(py)')
# @show isapprox(Array(ay), Array(cy))
# @show isapprox(a18[1](ax), c18[1](cx))
# @show isapprox(a18[1:2](ax), c18[1:2](cx))
# @show isapprox(a18[1:3](ax), c18[1:3](cx))
# @show isapprox(a18[1:4](ax), c18[1:4](cx))
# @show isapprox(a18[1:5](ax), c18[1:5](cx))


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
