using FileIO, ImageCore, ImageMagick, ImageTransformations, PyCall
import Knet

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

function imagenet_preprocess(img::Matrix{<:RGB}; normalization="torch", format="whcn", atype=Knet.atype)
    img = imresize(img, ratio=256/minimum(size(img)))           # min(h,w)=256
    hcenter,vcenter = size(img) .>> 1
    img = img[hcenter-111:hcenter+112, vcenter-111:vcenter+112] # h,w=224,224
    img = channelview(img)                                      # c,h,w=3,224,224
    if normalization == "tf"
        img = img .* 2 .- 1
    elseif normalization == "torch"
        μ,σ = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img = (img .- μ) ./ σ
    elseif normalization == "caffe"
        error("Caffe mode not implemented yet")
    else
        error("Unknown normalization: $normalization")
    end
    img = reshape(img, (1, size(img)...))                       # n,c,h,w=1,3,224,224
    perm = findfirst.(collect(format), "nchw")
    img = permutedims(img, (perm...,))
    return atype(img)
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

# Human readable predictions from tensors, images, files
function imagenet_predict(model, img; mode=nothing, normalization="torch", format="whcn", atype=Knet.atype)
    if mode == "torch"
        normalization, format = "torch", "nchw"
        atype = x->pyimport("torch").tensor(convert(Array{Float32},x))
    end
    img = imagenet_preprocess(img; normalization, format, atype)
    cls = model(img)
    if mode == "torch"; cls = cls.detach().cpu().numpy(); end
    cls = convert(Array, softmax(vec(cls)))
    idx = sortperm(cls, rev=true)
    [ idx cls[idx] imagenet_labels()[idx] ]
end

# Recursive walk and predict each image in a directory
function imagenet_walkdir(model, dir; n=typemax(Int), b=32, o...)
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
            images[1+k] = imagenet_preprocess(files[i+k]; o...)
        end
        batch = cat(images[1:j-i+1]...; dims=4)
        p = convert(Array, model(batch))
        append!(preds, vec((i->i[1]).(argmax(p; dims=1))))
    end
    [ preds imagenet_labels()[preds] files[1:n] ]
end


# Top-1 error in validation set
function imagenet_top1(model, valdir; o...)
    pred = imagenet_walkdir(model, valdir; o...)
    gold = imagenet_val()
    error = 0
    for i in 1:size(pred,1)
        image = match(r"ILSVRC2012_val_\d+", pred[i,3]).match
        if pred[i,1] != gold()[image]
            error += 1
        end
    end
    return error / size(pred,1)
end
