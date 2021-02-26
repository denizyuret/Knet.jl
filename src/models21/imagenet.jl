using FileIO, ImageCore, ImageMagick, ImageTransformations, PyCall, Artifacts, Base.Threads
import Knet

function torchvision_preprocess(file::String; atype=Knet.atype, resolution=224, format="whcn")
    transforms = pyimport("torchvision.transforms")
    torchvision_transform = transforms.Compose([
        transforms.Resize(resolution * 8 ÷ 7),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    PIL = pyimport("PIL")
    img = nothing
    @pywith PIL.Image.open(file) as f begin
        img = torchvision_transform(f.convert("RGB"))
    end
    img = img.numpy()
    img = reshape(img, (1, size(img)...)) # n,c,h,w=1,3,224,224
    if format != "nchw"
        perm = findfirst.(collect(format), "nchw")
        img = permutedims(img, (perm...,))
    end
    return atype(img)
end


# tf.keras.applications.imagenet_utils.preprocess_input
function imagenet_preprocess(file::String; o...)
    img = occursin(r"^http", file) ? mktemp() do fn,io
        load(download(file,fn))
    end : load(file)            # size=(h,w)
    imagenet_preprocess(img; o...)
end

function imagenet_preprocess(img::Matrix{<:Gray}; o...)
    imagenet_preprocess(RGB.(img); o...)
end

function imagenet_preprocess(img::Matrix{<:RGB}; mode="nothing", normalization="torch", format="whcn", atype=Knet.atype, resolution=224)
    if mode == "torch"
        normalization, format = "torch", "nchw"
        atype = x->pyimport("torch").tensor(convert(Array{Float32},x))
    end
    if mode == "tf"
        normalization, format, atype = "tf", "nhwc", Array{Float32}
    end
    minsize = resolution * 8 ÷ 7
    img = imresize(img, ratio=minsize/minimum(size(img)))           # min(h,w)=256
    hcenter,vcenter = size(img) .>> 1
    half = resolution ÷ 2
    img = img[hcenter-half+1:hcenter+half, vcenter-half+1:vcenter+half] # h,w=224,224
    img = channelview(img)                                      # c,h,w=3,224,224
    img = Float32.(img)
    if normalization == "tf"
        xmin, xmax = extrema(img)
        img = img .* (2/(xmax-xmin)) .- ((xmax+xmin)/(xmax-xmin))
    elseif normalization == "torch"
        μ,σ = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img = (img .- μ) ./ σ
    elseif normalization == "caffe"
        error("Caffe mode not implemented yet")
    elseif normalization === nothing
        # do nothing
    else
        error("Unknown normalization: $normalization")
    end
    img = reshape(img, (1, size(img)...))                       # n,c,h,w=1,3,224,224
    perm = findfirst.(collect(format), "nchw")
    img = permutedims(img, (perm...,))
    return atype(img)
end

# fallback
imagenet_preprocess(x; o...) = x

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

# Human readable predictions from tensors, images, files
function imagenet_predict(model, img="ILSVRC2012_val_00000001.JPEG"; preprocess=identity, apply_softmax=true)
    img = preprocess(img)
    cls = model(img)
    if cls isa PyObject; cls = cls.cpu().numpy(); end
    cls = convert(Array, vec(cls))
    cls = apply_softmax ? softmax(cls) : cls
    idx = sortperm(cls, rev=true)
    [ idx cls[idx] imagenet_labels()[idx] ]
end

# Recursive walk and predict each image in a directory
function imagenet_walkdir(
    model, dir;
    n=typemax(Int),
    b=32,
    preprocess=imagenet_preprocess,
    format="whcn",
    batchinput = identity,
    batchoutput = (x->convert(Array,x)),
    usethreads = true,
    o...
)
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
        if usethreads
            @threads for k in 0:j-i
                images[1+k] = preprocess(files[i+k]; format, o...)
            end
        else
            for k in 0:j-i
                images[1+k] = preprocess(files[i+k]; format, o...)
            end
        end
        batch = cat(images[1:j-i+1]...; dims=findfirst('n', format))
        p = (batch |> batchinput |> model |> batchoutput)
        append!(preds, vec((i->i[1]).(argmax(p; dims=1))))
    end
    [ preds imagenet_labels()[preds] files[1:n] ]
end


# Top-1 error in validation set
function imagenet_top1(model, valdir; o...)
    pred = imagenet_walkdir(model, valdir; o...)
    gold = imagenet_val()
    error, count = 0, size(pred, 1)
    for i in 1:count
        image = match(r"ILSVRC2012_val_\d+", pred[i,3]).match
        if pred[i,1] != gold[image]
            error += 1
        end
    end
    error = error / count
    (accuracy = 1-error, error = error)
end


# torch model top-1 example: 
# r18a = models.resnet18(pretrained=true).eval()
# imagenet_top1(r18a, val; n=1000, preprocess=torchvision_preprocess, format="nchw", batchinput=torch.tensor, batchoutput=(x->permutedims(x.detach().cpu().numpy())), atype=Array{Float32}, usethreads=false)
