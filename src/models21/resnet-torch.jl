include("resnet.jl")

using Knet.Train20: param
using Knet.KnetArrays: KnetArray
using CUDA: CuArray
using PyCall, AutoGrad
import NNlib, Knet

torch = pyimport("torch")
nn = pyimport("torch.nn")
models = pyimport("torchvision.models")
t2a(x) = x.cpu().detach().numpy()
Base.isapprox(a::Array,b::CuArray)=isapprox(a,Array(b))
Base.isapprox(a::CuArray,b::Array)=isapprox(Array(a),b)
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
    conv1 = Conv(p.conv1; normalization=bn1, activation=relu)
    conv2 = Conv(p.conv2; normalization=bn2)
    f1 = Sequential(conv1, conv2)
    if p.downsample === nothing
        f2 = identity
    else
        bn3 = BatchNorm(p.downsample[2])
        f2 = Conv(p.downsample[1]; normalization=bn3)
    end
    return Residual(f1, f2; activation=relu)
end


function ResNetInput(conv1::PyObject, bn1::PyObject)
    bn1 = BatchNorm(bn1)
    Sequential(
        Conv(conv1; normalization=bn1, activation=relu),
        x->pool(x; window=3, stride=2, padding=1);
        name = "Input"
    )
end


function ResNetOutput(fc::PyObject)
    w = param(t2a(fc.weight))
    bias = param(t2a(fc.bias))
    Sequential(
        x->pool(x; mode=1, window=(size(x,1),size(x,2))),
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
        ; use_estimates = true, #DBG
        update = 0, #DBG b.momentum,
        mean = bnweight(b.running_mean).value, #DBG .value,
        var = bnweight(b.running_var).value, #DBG .value,
        bias = bnweight(b.bias),
        scale = bnweight(b.weight),
        epsilon = b.eps,
    )
end

T = Float64

#isfile("dog.jpg") || download("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
download("https://www.ilikeorchids.com/media/qlcdv53x/iap-levoplant-low-res-141.jpg?center=0.47460844803037494,0.54166666666666663&mode=crop&width=960&height=1080&rnd=132344581373430000","dog.jpg")
isfile("imagenet_classes.txt") || download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
classes = readlines("imagenet_classes.txt")

Image = pyimport("PIL.Image")
transforms = pyimport("torchvision.transforms")
input_image = Image.open("dog.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

p18 = models.resnet18(pretrained=true).eval()
px = t2a(input_batch)
py = t2a(p18(torch.tensor(px)))

Knet.atype() = Array{T}
a18 = ResNetBasic(p18)
ax = Knet.atype(permutedims(px,(4,3,2,1)))
ay = a18(ax)
@show isapprox(Array(ay), Array(py)')
ap = Param(ax)
@show @gcheck a18(ap) (nsample=3,)

Knet.atype() = KnetArray{T}
k18 = ResNetBasic(p18)
@show chkparams(k18,a18)
kx = Knet.atype(permutedims(px,(4,3,2,1)))
ky = k18(kx)
@show isapprox(Array(ay), Array(ky))
kp = Param(kx)
@show @gcheck k18(kp) (nsample=3,)

Knet.atype() = CuArray{T}
c18 = ResNetBasic(p18)
@show chkparams(c18,a18)
cx = Knet.atype(permutedims(px,(4,3,2,1)))
cy = c18(cx)
@show isapprox(Array(ay), Array(cy))
cp = Param(cx)
@show @gcheck c18(cp) (nsample=3,)

nothing
