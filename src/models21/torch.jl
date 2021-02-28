include("Models21.jl")
include("imagenet.jl")
import Knet
using Knet.Layers21, Knet.Ops21
using Knet.Train20: param
using Knet.Ops20: pool, softmax
using Knet.KnetArrays: KnetArray
using CUDA: CuArray
using PyCall, AutoGrad, SHA, Tar, DelimitedFiles
using ImageCore, ImageTransformations, FileIO, Artifacts, Base.Threads, Base.Iterators

torch = pyimport("torch")
nn = pyimport("torch.nn")
models = pyimport("torchvision.models")
transforms = pyimport("torchvision.transforms")
PIL = pyimport("PIL")

t2a(x) = x.cpu().detach().numpy()
chkparams(a,b)=((pa,pb)=params.((a,b)); length(pa)==length(pb) && all(isapprox.(pa,pb)))
rdims(a)=permutedims(a, ((ndims(a):-1:1)...,))
typename(p::PyObject) = pytypeof(p).__name__


function ResNetTorch(p::PyObject)
    layers = (p.layer1, p.layer2, p.layer3, p.layer4)
    s = Sequential(ResNetInput(p.conv1, p.bn1); name="ResNet$(length.(layers))")
    for (i, layer) in enumerate(layers)
        blocks = Sequential(; name="Layer$i")
        for block in layer
            push!(blocks, ResNetBlock(block))
        end
        push!(s, blocks)
    end
    push!(s, ResNetOutput(p.fc))
end


function ResNetBlock(p::PyObject)
    i, layers = 1, []
    while haskey(p, "conv$i")
        convi = getproperty(p, "conv$i")
        bni = getproperty(p, "bn$i")
        push!(layers, Conv2d(convi; normalization=BatchNorm2d(bni), activation=relu))
        i += 1
    end
    layers[end].activation = nothing
    r = (p.downsample === nothing ? identity :
         Conv2d(p.downsample[1]; normalization=BatchNorm2d(p.downsample[2])))
    Residual(Sequential(layers...), r; activation=relu)
end


function ResNetInput(conv1::PyObject, bn1::PyObject)
    bn1 = BatchNorm2d(bn1)
    Sequential(
        Conv2d(conv1; normalization=bn1, activation=relu),
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
        Linear(w; bias);
        name = "Output"
    )
end


function Conv2d(p::PyObject; normalization=nothing, activation=nothing)
    w = param(permutedims(t2a(p.weight), (4,3,2,1)))
    bias = (p.bias === nothing ? nothing :
            param(reshape(t2a(p.bias), (1,1,:,1))))
    Conv(w; bias, normalization, activation,
         p.padding, p.stride, p.dilation, p.groups)
end


function BatchNorm2d(b::PyObject)
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


function BatchNorm2d2(b::PyObject) # use this for @gcheck, defaults will be fixed when weights loaded from file
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


function resnetimport(model)
    # saves pytorch model to e.g. /home/dyuret/.cache/torch/hub/checkpoints/resnext50_32x4d-7cdf4587.pth
    @assert haskey(models, model) 
    pm = getproperty(models, model)(pretrained=true).eval()
    px = randn(Float32, 224, 224, 3, 1)
    py = px |> rdims |> torch.tensor |> pm |> t2a |> rdims

    T = Float32 ## use Float64 for @gcheck

    save_type = Knet.array_type[]
    Knet.array_type[] = Array{T}
    am = ResNetTorch(pm)
    ax = Knet.atype(px)
    ay = am(ax)
    @show ay ≈ py
    #ap = Param(ax)
    #@show @gcheck am(ap) (nsample=3,)

    Knet.array_type[] = KnetArray{T}
    km = ResNetTorch(pm)
    @show chkparams(km,am)
    kx = Knet.atype(px)
    ky = km(kx)
    @show ky ≈ py
    #kp = Param(kx)
    #@show @gcheck km(kp) (nsample=3,)

    Knet.array_type[] = CuArray{T}
    cm = ResNetTorch(pm)
    @show chkparams(cm,am)
    cx = Knet.atype(px)
    cy = cm(cx)
    @show cy ≈ py
    #cp = Param(cx)
    #@show @gcheck cm(cp) (nsample=3,)

    @info "Saving $(model).jld2"
    saveweights("$(model).jld2", cm; atype=CuArray)

    @info "Loading $(model).jld2"
    setweights!(cm, "$(model).jld2")
    @show chkparams(cm,am)
    cx = Knet.atype(px)
    cy = cm(cx)
    @show cy ≈ py
    #cp = Param(cx)
    #@show @gcheck cm(cp) (nsample=3,)

    run(`tar cf $(model).tar $(model).jld2`)
    sha1 = Tar.tree_hash("$(model).tar")
    @info "git-tree-sha1 = \"$sha1\""

    run(`gzip $(model).tar`)
    sha2 = open("$(model).tar.gz") do f; bytes2hex(sha256(f)); end
    @info "sha256 = \"$sha2\""

    Knet.array_type[] = save_type
    am
end


# Preprocessing for pretrained models:
resnetinput(x) = Knet.atype(x)

function resnetinput(file::String)
    img = occursin(r"^http", file) ? mktemp() do fn,io
        load(download(file,fn))
    end : load(file)
    resnetinput(img)
end


function resnetinput(img::Matrix{<:Gray})
    resnetinput(RGB.(img))
end


function resnetinput(img::Matrix{<:RGB})
    img = imresize(img, ratio=256/minimum(size(img))) # min(h,w)=256
    hcenter,vcenter = size(img) .>> 1
    img = img[hcenter-111:hcenter+112, vcenter-111:vcenter+112] # h,w=224,224
    img = channelview(img)                                      # c,h,w=3,224,224
    μ,σ = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img = (img .- μ) ./ σ
    img = permutedims(img, (3,2,1)) # 224,224,3
    img = reshape(img, (size(img)..., 1)) # 224,224,3,1
    Knet.atype(img)
end


function resnetinput_python(file::String) # note that this crashes when used with @threads
    global _resinput_python
    if !(@isdefined _resinput_python)
        _resinput_python = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    end
    input_tensor = nothing
    @pywith PIL.Image.open(file) as input_image begin
        input_tensor = _resinput_python(input_image.convert("RGB"))
    end
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return Knet.atype(permutedims(t2a(input_batch), (4,3,2,1)))
end


# Human readable predictions from tensors, images, files
function resnetpred(model, img)
    cls = convert(Array, softmax(vec(model(resnetinput(img)))))
    idx = sortperm(cls, rev=true)
    [ idx cls[idx] imagenet_labels()[idx] ]
end


# Recursive walk and predict each image in a directory
function resnetdir(model, dir; n=typemax(Int), b=32)
    files = []
    for (root, dirs, fs) in walkdir(dir)
        for f in fs
            push!(files, joinpath(root, f))
            length(files) > n && break
        end
        length(files) > n && break
    end
    n = min(n, length(files))
    images = Array{Any}(undef, b)
    preds = []
    for i in Knet.progress(1:b:n)
        j = min(n, i+b-1)
        @threads for k in 0:j-i
            images[1+k] = resnetinput(files[i+k])
        end
        batch = cat(images[1:j-i+1]...; dims=4)
        p = convert(Array, model(batch))
        append!(preds, vec((i->i[1]).(argmax(p; dims=1))))
    end
    [ preds imagenet_labels()[preds] files[1:n] ]
end


# Top-1 error in validation set
function resnettop1(model, valdir; o...)
    pred = resnetdir(model, valdir; o...)
    error = 0
    for i in 1:size(pred,1)
        image = match(r"ILSVRC2012_val_\d+", pred[i,3]).match
        if pred[i,1] != imagenet_val()[image]
            error += 1
        end
    end
    error = error / size(pred,1)
    (accuracy = 1-error, error = error)
end


##### MobileNetV2 ##################################################################

function torch2knet(p::PyObject; o...)
    converter = Symbol(pytypeof(p).__name__)
    if isdefined(@__MODULE__, converter)
        eval(converter)(p; o...)
    else
        @warn "$converter not defined"
    end
end

function adaptive_avg_pool2d(x)
    y = pool(x; mode=1, window=size(x)[1:2])
    reshape(y, size(y,3), size(y,4))
end

function MobileNetV2(p::PyObject)
    s = Sequential()
    for l in p.features
        push!(s, torch2knet(l))
    end
    push!(s, adaptive_avg_pool2d)
    dropout = p.classifier[1].p
    push!(s, Linear(p.classifier[2]; dropout))
    return s
end

function Dropout(p::PyObject)
    Op(dropout, p.p)
end

function Linear(p::PyObject; dropout=0, activation=nothing)
    w = param(t2a(p.weight))
    bias = p.bias === nothing ? nothing : param(t2a(p.bias))
    Linear(w; bias, dropout, activation)
end

function ConvBNReLU(p::PyObject)
    activation = torch2knet(p[3])
    normalization = torch2knet(p[2])
    Conv2d(p[1]; activation, normalization)
end

function ReLU6(p::PyObject)
    Op(relu; max_value=6)
end

function InvertedResidual(p::PyObject)
    s = Sequential()
    for l in p.conv
        if l == p.conv[end]
            @assert typename(l) == "BatchNorm2d"
            s[end].normalization = torch2knet(l)
        else
            push!(s, torch2knet(l))
        end
    end
    if s[2].stride == (1,1) && size(s[1].w,3) == size(s[end].w,4)
        return Residual(s)
    else
        return s
    end
end


function mobilenet2import()
    # saves pytorch model to e.g. /home/dyuret/.cache/torch/hub/checkpoints/resnext50_32x4d-7cdf4587.pth
    model = "mobilenet_v2_100_224_pt"
    pm = models.mobilenet_v2(pretrained=true).eval()
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

    @info "Saving $(model).jld2"
    saveweights("$(model).jld2", cm; atype=CuArray)

    @info "Loading $(model).jld2"
    dm = MobileNet(model; pretrained=false)
    setweights!(dm, "$(model).jld2")
    @show chkparams(dm,am)
    dx = Knet.atype(px)
    dy = dm(dx)
    @show dy ≈ py
    #dp = Param(dx)
    #@show @gcheck dm(dp) (nsample=3,)

    run(`tar cf $(model).tar $(model).jld2`)
    sha1 = Tar.tree_hash("$(model).tar")
    @info "git-tree-sha1 = \"$sha1\""

    run(`gzip $(model).tar`)
    sha2 = open("$(model).tar.gz") do f; bytes2hex(sha256(f)); end
    @info "sha256 = \"$sha2\""

    Knet.array_type[] = save_type
    am
end
