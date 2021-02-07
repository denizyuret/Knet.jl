include("Models21.jl")
using Knet.Train20: param
using Knet.KnetArrays: KnetArray
using CUDA: CuArray
using PyCall, AutoGrad, SHA, Tar, DelimitedFiles
import NNlib, Knet, Tar


function ResNetTorch(p::PyObject)
    layers = (p.layer1, p.layer2, p.layer3, p.layer4)
    s = Sequential(ResNetInput(p.conv1, p.bn1); name="ResNet$(length.(layers))")
    for (i, layer) in enumerate(layers)
        blocks = Sequential(; name="Layer$layer")
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


torch = pyimport("torch")
nn = pyimport("torch.nn")
models = pyimport("torchvision.models")
t2a(x) = x.cpu().detach().numpy()
chkparams(a,b)=((pa,pb)=params.((a,b)); length(pa)==length(pb) && all(isapprox.(pa,pb)))
rdims(a)=permutedims(a, ((ndims(a):-1:1)...,))


function resnetsave(model)
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


# https://www.adeveloperdiary.com/data-science/computer-vision/how-to-prepare-imagenet-dataset-for-image-classification/

function resneteval()
    global imagenet_classes, map_clsloc
    imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    map_clsloc_url = ""
    @isdefined(imagenet_classes) || mktemp() do fn,io
        imagenet_classes = readlines(download(imagenet_classes_url,fn))
    end
    @isdefined(map_clsloc) || mktemp do fn, io
        map_clsloc = readdlm(download(map_clsloc_url, fn))
    end
end

  
#=
### Python preprocess
isfile("dog.jpg") || download("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
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
px = t2a(input_batch)

### Julia preprocess
using Images, FileIO
j1 = load("dog.jpg")
j2 = imresize(j1, ratio=256/minimum(size(j1)))
h,w = size(j2) .÷ 2
j3 = j2[h-111:h+112, w-111:w+112] # h,w=224,224
j4 = T.(channelview(j3))                  # c,h,w=3,224,224
jmean=reshape([0.485, 0.456, 0.406], (3,1,1))
jstd=reshape([0.229, 0.224, 0.225], (3,1,1))
j5 = (j4 .- jmean) ./ jstd
j6 = permutedims(reshape(j5,(1,size(j5)...)),(4,3,2,1))

=#
