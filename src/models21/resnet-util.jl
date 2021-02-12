include("Models21.jl")
import Knet
using Knet.Layers21, Knet.Ops21
using Knet.Train20: param
using Knet.Ops20: pool, softmax
using Knet.KnetArrays: KnetArray
using CUDA: CuArray
using PyCall, AutoGrad, SHA, Tar, DelimitedFiles
using ImageCore, ImageTransformations, FileIO, Artifacts, Base.Threads

torch = pyimport("torch")
nn = pyimport("torch.nn")
models = pyimport("torchvision.models")
transforms = pyimport("torchvision.transforms")
PIL = pyimport("PIL")

t2a(x) = x.cpu().detach().numpy()
chkparams(a,b)=((pa,pb)=params.((a,b)); length(pa)==length(pb) && all(isapprox.(pa,pb)))
rdims(a)=permutedims(a, ((ndims(a):-1:1)...,))


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
        push!(layers, Conv(convi; normalization=BatchNorm(bni), activation=relu))
        i += 1
    end
    layers[end].activation = nothing
    r = (p.downsample === nothing ? identity :
         Conv(p.downsample[1]; normalization=BatchNorm(p.downsample[2])))
    Residual(Sequential(layers...), r; activation=relu)
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
        Linear(w; bias);
        name = "Output"
    )
end


function Conv(p::PyObject; normalization=nothing, activation=nothing)
    w = param(permutedims(t2a(p.weight), (4,3,2,1)))
    bias = (p.bias === nothing ? nothing :
            param(reshape(t2a(p.bias), (1,1,:,1))))
    Conv(w; bias, normalization, activation,
         p.padding, p.stride, p.dilation, p.groups)
end


function BatchNorm(b::PyObject)
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


function BatchNorm2(b::PyObject) # use this for @gcheck, defaults will be fixed when weights loaded from file
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
    return error / size(pred,1)
end


# ImageNet meta-information
function imagenet_labels()
    global _imagenet_labels
    if !@isdefined(_imagenet_labels)
        _imagenet_labels = [ replace(x, r"\S+ ([^,]+).*"=>s"\1") for x in
                             readlines(joinpath(artifact"imagenet_labels","LOC_synset_mapping.txt")) ]
    end
    _imagenet_labels
end

function imagenet_synsets()
    global _imagenet_synsets
    if !@isdefined(_imagenet_synsets)
        _imagenet_synsets = [ split(s)[1] for s in
                              readlines(joinpath(artifact"imagenet_labels", "LOC_synset_mapping.txt")) ]
    end
    _imagenet_synsets
end

function imagenet_val()
    global _imagenet_val
    if !@isdefined(_imagenet_val)
        synset2index = Dict(s=>i for (i,s) in enumerate(imagenet_synsets()))
        _imagenet_val = Dict(x=>synset2index[y] for (x,y) in (z->split(z,[',',' '])[1:2]).(
            readlines(joinpath(artifact"imagenet_labels", "LOC_val_solution.csv"))[2:end]))
    end
    _imagenet_val
end
