include("../Models21.jl")
import Knet
using Knet.Layers21, Knet.Ops21
using Knet.Train20: param
using Knet.Ops20: pool, softmax
using Knet.KnetArrays: KnetArray
using CUDA: CuArray
using PyCall, AutoGrad, SHA, Tar, DelimitedFiles
using ImageCore, ImageTransformations, FileIO, Artifacts, Base.Threads, Base.Iterators

# pytorch models are saved in e.g. /home/dyuret/.cache/torch/hub/checkpoints/resnext50_32x4d-7cdf4587.pth
torch = pyimport("torch")
nn = pyimport("torch.nn")
models = pyimport("torchvision.models")
transforms = pyimport("torchvision.transforms")
PIL = pyimport("PIL")

t2a(x) = x.cpu().detach().numpy()
chkparams(a,b)=((pa,pb)=params.((a,b)); length(pa)==length(pb) && all(isapprox.(pa,pb)))
rdims(a)=permutedims(a, ((ndims(a):-1:1)...,))
typename(p::PyObject) = pytypeof(p).__name__


"""
    torchimport("resnet18", "resnet18", ResNet)
    torchimport("mobilenet_v2", "mobilenet_v2_100_224_pt", MobileNet)
"""
function torchimport(model, file=nothing, testmodel=nothing)
    @assert haskey(models, model) 
    pm = getproperty(models,model)(pretrained=true).eval()
    px = randn(Float32, 224, 224, 3, 1)
    py = px |> rdims |> torch.tensor |> pm |> t2a |> rdims

    T = Float32 ## use Float64 for @gcheck

    save_type = Knet.array_type[]
    Knet.array_type[] = Array{T}
    am = torch2knet(pm)
    ax = Knet.atype(px)
    ay = am(ax)
    @show ay ≈ py
    #ap = Param(ax)
    #@show @gcheck am(ap) (nsample=3,)

    Knet.array_type[] = KnetArray{T}
    km = torch2knet(pm)
    @show chkparams(km,am)
    kx = Knet.atype(px)
    ky = km(kx)
    @show ky ≈ py
    #kp = Param(kx)
    #@show @gcheck km(kp) (nsample=3,)

    Knet.array_type[] = CuArray{T}
    cm = torch2knet(pm)
    @show chkparams(cm,am)
    cx = Knet.atype(px)
    cy = cm(cx)
    @show cy ≈ py
    #cp = Param(cx)
    #@show @gcheck cm(cp) (nsample=3,)

    if file !== nothing
        @info "Saving $(file).jld2"
        saveweights("$(file).jld2", cm; atype=CuArray)

        if testmodel !== nothing
            @info "Loading $(file).jld2"
            dm = testmodel(file; pretrained=false)
            setweights!(dm, "$(file).jld2")
            @show chkparams(dm,am)
            dx = Knet.atype(px)
            dy = dm(dx)
            @show dy ≈ py
            #dp = Param(dx)
            #@show @gcheck dm(dp) (nsample=3,)
        end

        run(`tar cf $(file).tar $(file).jld2`)
        sha1 = Tar.tree_hash("$(file).tar")
        @info "git-tree-sha1 = \"$sha1\""

        run(`gzip $(file).tar`)
        sha2 = open("$(file).tar.gz") do f; bytes2hex(sha256(f)); end
        @info "sha256 = \"$sha2\""
    end

    Knet.array_type[] = save_type
    am
end


function torch2knet(p::PyObject; o...)
    converter = Symbol(pytypeof(p).__name__ * "_pt")
    if isdefined(@__MODULE__, converter)
        eval(converter)(p; o...)
    else
        @warn "$converter not defined"
    end
end


### ResNet

function ResNet_pt(p::PyObject)
    layers = (p.layer1, p.layer2, p.layer3, p.layer4)
    s = Sequential(ResNetInput_pt(p.conv1, p.bn1); name="ResNet$(length.(layers))")
    for (i, layer) in enumerate(layers)
        blocks = Sequential(; name="Layer$i")
        for block in layer
            push!(blocks, ResNetBlock_pt(block))
        end
        push!(s, blocks)
    end
    push!(s, ResNetOutput_pt(p.fc))
end


function ResNetBlock_pt(p::PyObject)
    i, layers = 1, []
    while haskey(p, "conv$i")
        convi = getproperty(p, "conv$i")
        bni = getproperty(p, "bn$i")
        push!(layers, Conv2d_pt(convi; normalization=torch2knet(bni), activation=relu))
        i += 1
    end
    layers[end].activation = nothing
    r = (p.downsample === nothing ? identity :
         Conv2d_pt(p.downsample[1]; normalization=torch2knet(p.downsample[2])))
    Residual(Sequential(layers...), r; activation=relu)
end


function ResNetInput_pt(conv1::PyObject, bn1::PyObject)
    Sequential(
        Conv2d_pt(conv1; normalization=torch2knet(bn1), activation=relu),
        x->pool(x; window=3, stride=2, padding=1);
        name = "Input"
    )
end


function ResNetOutput_pt(fc::PyObject)
    w = param(t2a(fc.weight))
    bias = param(t2a(fc.bias))
    Sequential(
        x->pool(x; mode=1, window=size(x)[1:end-2]),
        x->reshape(x, :, size(x,4)),
        Linear(w; bias);
        name = "Output"
    )
end


function Conv2d_pt(p::PyObject; normalization=nothing, activation=nothing)
    w = param(permutedims(t2a(p.weight), (4,3,2,1)))
    bias = (p.bias === nothing ? nothing :
            param(reshape(t2a(p.bias), (1,1,:,1))))
    Conv(w; bias, normalization, activation,
         p.padding, p.stride, p.dilation, p.groups)
end


function BatchNorm2d_pt(b::PyObject)
    bnweight(x) = param(reshape(t2a(x), (1,1,:,1)))
    BatchNorm(
        ; use_estimates = nothing,
        update = b.momentum,
        mean = bnweight(b.running_mean).value,
        var = bnweight(b.running_var).value,
        bias = bnweight(b.bias),
        scale = bnweight(b.weight),
        epsilon = b.eps,
    )
end


function BatchNorm2d_gcheck(b::PyObject) # use this for @gcheck, defaults will be fixed when weights loaded from file
    bnweight(x) = param(reshape(t2a(x), (1,1,:,1)))
    BatchNorm(
        ; use_estimates = true, ### nothing,
        update = 0, ### b.momentum,
        mean = bnweight(b.running_mean), ### .value,
        var = bnweight(b.running_var), ### .value,
        bias = bnweight(b.bias),
        scale = bnweight(b.weight),
        epsilon = b.eps,
    )
end


##### MobileNetV2 ##################################################################

function adaptive_avg_pool2d(x)
    y = pool(x; mode=1, window=size(x)[1:end-2])
    reshape(y, size(y,3), size(y,4))
end

function MobileNetV2_pt(p::PyObject)
    s = Sequential()
    for l in p.features
        push!(s, torch2knet(l))
    end
    push!(s, adaptive_avg_pool2d)
    dropout = p.classifier[1].p
    push!(s, torch2knet(p.classifier[2]; dropout))
    return s
end

function MobileNetV3_pt(p::PyObject)
    s = Sequential()
    for l in p.features
        push!(s, torch2knet(l))
    end
    push!(s, torch2knet(p.avgpool))
    push!(s, torch2knet(p.classifier[1]; activation=torch2knet(p.classifier[2])))
    dropout = p.classifier[3].p
    push!(s, torch2knet(p.classifier[4]; dropout))
    return s
end

function Dropout_pt(p::PyObject)
    Op(dropout, p.p)
end

function Linear_pt(p::PyObject; dropout=0, activation=nothing)
    w = param(t2a(p.weight))
    bias = p.bias === nothing ? nothing : param(t2a(p.bias))
    Linear(w; bias, dropout, activation)
end

function ConvBNReLU_pt(p::PyObject)
    activation = torch2knet(p[3])
    normalization = torch2knet(p[2])
    Conv2d_pt(p[1]; activation, normalization)
end

function ReLU6_pt(p::PyObject)
    Op(relu; max_value=6)
end

function InvertedResidual_pt(p::PyObject)
    haskey(p, :conv) ? InvertedResidualV2(p) : InvertedResidualV3(p)
end

function InvertedResidualV2(p::PyObject)
    s = Sequential()
    for l in p.conv
        if l == p.conv[end]
            @assert typename(l) == "BatchNorm2d"
            s[end].normalization = torch2knet(l)
        else
            push!(s, torch2knet(l))
        end
    end
    if p.use_res_connect # s[2].stride == (1,1) && size(s[1].w,3) == size(s[end].w,4)
        return Residual(s)
    else
        return s
    end
end

function InvertedResidualV3(p::PyObject)
    s = Sequential()
    for l in p.block
        push!(s, torch2knet(l))
    end
    if p.use_res_connect # s[2].stride == (1,1) && size(s[1].w,3) == size(s[end].w,4)
        return Residual(s)
    else
        return s
    end
end

function SqueezeExcitation_pt(p::PyObject)
    SqueezeExcitation(
        torch2knet(p.fc1; activation=relu),
        torch2knet(p.fc2; activation=hardsigmoid),
    )
end

function ConvBNActivation_pt(p::PyObject)
    activation = torch2knet(p[3])
    normalization = torch2knet(p[2])
    torch2knet(p[1]; activation, normalization)
end

function AdaptiveAvgPool2d_pt(p::PyObject)
    return adaptive_avg_pool2d
end

function Hardswish_pt(p::PyObject)
    return hardswish
end

function ReLU_pt(p::PyObject)
    return relu
end

function Identity_pt(p::PyObject)
    return identity
end


