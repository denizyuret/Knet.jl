import Knet
using FileIO, ImageCore, ImageMagick, ImageTransformations


# main
function imagenet_preprocess(img::Matrix{<:RGB}; normalize=nothing, resolution=224, format="whcn", atype=Knet.atype)
    if normalize === nothing; normalize = identity; end
    minsize = resolution * 8 ÷ 7                          # 224 => 256
    img = imresize(img, ratio=minsize/minimum(size(img))) # min(h,w)=256
    hcenter,vcenter = size(img) .>> 1                     # h÷2, w÷2
    half = resolution ÷ 2
    img = img[hcenter-half+1:hcenter+half, vcenter-half+1:vcenter+half] # hw=224,224
    img = Float32.(channelview(img))                                    # chw=3,224,224
    img = normalize(img)
    img = reshape(img, (1, size(img)...)) # nchw=1,3,224,224
    if format != "nchw"
        perm = findfirst.(collect(format), "nchw")
        img = permutedims(img, (perm...,))
    end
    return atype(img)
end


# fallback for arbitrary array types
imagenet_preprocess(x; atype=Knet.atype, o...) = atype(x)


# grayscale images
function imagenet_preprocess(img::Matrix{<:Gray}; o...)
    imagenet_preprocess(RGB.(img); o...)
end


# files and urls
function imagenet_preprocess(file::String; o...)
    if occursin(r"^https?://", file)
        mktemp() do fn,io
            imagenet_preprocess(load(download(url,fn)); o...)
        end
    else
        imagenet_preprocess(load(file); o...)
    end
end
