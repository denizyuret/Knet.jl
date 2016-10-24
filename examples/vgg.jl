for p in ("Knet","ArgParse","ImageMagick","MAT")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""

julia vgg.jl image-file-or-url

This example implements the VGG model from `Very Deep Convolutional
Networks for Large-Scale Image Recognition', Karen Simonyan and Andrew
Zisserman, arXiv technical report 1409.1556, 2014.  In particular we
use the 16 layer network, denoted as configuration D in the technical
report.

* Paper url: https://arxiv.org/abs/1409.1556
* Project page: http://www.robots.ox.ac.uk/~vgg/research/very_deep
* MatConvNet weights used here: http://www.vlfeat.org/matconvnet/pretrained

"""
module VGG
using Knet,ArgParse,Images,MAT
const imgurl = "https://github.com/BVLC/caffe/raw/master/examples/images/cat.jpg"
const vggurl = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat"

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="vgg.jl (c) Deniz Yuret, 2016. Classifying images with the VGG model from http://www.robots.ox.ac.uk/~vgg/research/very_deep."
    # s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("image"; required=true; help="Image file or URL.")
        ("--model"; default=Knet.dir("data","imagenet-vgg-verydeep-16.mat"); help="Location of the model file")
        ("--top"; default=5; arg_type=Int; help="Display the top N classes")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    gpu() >= 0 || error("VGG only works on GPU machines.")
    if !isfile(o[:model])
        println("Should I download the VGG model (492MB)? Enter 'y' to download, anything else to quit.")
        readline() == "y\n" || return
        download(vggurl,o[:model])
    end
    info("Reading $(o[:model])")
    vgg = matread(o[:model])
    w = weights(vgg["layers"])
    averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])
    description = vgg["meta"]["classes"]["description"]
    info("Reading $(o[:image])")
    x1 = data(o[:image], averageImage)
    info("Classifying")
    @time y1 = predict(w,x1)
    z1 = vec(Array(y1))
    s1 = sortperm(z1,rev=true)
    p1 = exp(logp(z1))
    display(hcat(p1[s1[1:o[:top]]], description[s1[1:o[:top]]]))
    println()
end

function data(img, averageImage)
    if contains(img,"://")
        info("Downloading $img")
        img = download(img)
    end
    a0 = load(img)
    new_size = ntuple(i->div(size(a0,i)*224,minimum(size(a0))),2)
    a1 = Images.imresize(a0, new_size)
    i1 = div(size(a1,1)-224,2)
    j1 = div(size(a1,2)-224,2)
    b1 = a1[i1+1:i1+224,j1+1:j1+224]
    c1 = separate(b1)
    d1 = convert(Array{Float32}, c1)
    e1 = reshape(d1[:,:,1:3], (224,224,3,1))
    f1 = (255 * e1 .- averageImage)
    g1 = permutedims(f1, [2,1,3,4])
    x1 = KnetArray(g1)
end

function weights(layers)
    w = Any[]
    for l in layers
        haskey(l,"weights") && !isempty(l["weights"]) && push!(w, l["weights"]...)
    end
    for i in 2:2:26
        w[i] = reshape(w[i], (1,1,length(w[i]),1))
    end
    for i in 27:2:32
        w[i] = mat(w[i])'
    end
    w = map(KnetArray,w)
end

const op = [1,2,1,2,1,1,2,1,1,2,1,1,2,3,3,4]

convx(w,x)=conv4(w,x;padding=1,mode=1)

function predict(w,x)
    for k=1:div(length(w),2)
        if op[k] == 1
            x = relu(convx(w[2k-1],x) .+ w[2k])
        elseif op[k] == 2
            x = pool(relu(convx(w[2k-1],x) .+ w[2k]))
        elseif op[k] == 3
            x = relu(w[2k-1]*mat(x) .+ w[2k])
        else
            x = w[2k-1]*mat(x) .+ w[2k]
        end
    end
    return x
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia vgg.jl cat.jpg
# julia> VGG.main("cat.jpg")
!isinteractive() && (!isdefined(Main,:load_only) || !Main.load_only) && main(ARGS)

end # module
