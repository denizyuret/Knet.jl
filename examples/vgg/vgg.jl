for p in ("Knet","ArgParse","Images","MAT","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""

julia vgg.jl image-file-or-url

This example implements the VGG model from `Very Deep Convolutional
Networks for Large-Scale Image Recognition', Karen Simonyan and Andrew
Zisserman, arXiv technical report 1409.1556, 2014. This example works
for D and E models currently. VGG-D is the default model if you do not
specify any model.

* Paper url: https://arxiv.org/abs/1409.1556
* Project page: http://www.robots.ox.ac.uk/~vgg/research/very_deep
* MatConvNet weights used here: http://www.vlfeat.org/matconvnet/pretrained

"""
module VGG
using Knet,ArgParse,Images,MAT,Compat
const imgurl = "https://github.com/BVLC/caffe/raw/master/examples/images/cat.jpg"
const vggurl = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat"
const LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="vgg.jl (c) Deniz Yuret, 2016. Classifying images with the VGG model from http://www.robots.ox.ac.uk/~vgg/research/very_deep."
    # s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("image"; default=imgurl; help="Image file or URL.")
        ("--model"; default=Knet.dir("data","imagenet-vgg-verydeep-16.mat"); help="Location of the model file")
        ("--top"; default=5; arg_type=Int; help="Display the top N classes")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array and float type to use")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    atype = eval(parse(o[:atype]))
    if !isfile(o[:model])
        println("Should I download the VGG model (492MB)? Enter 'y' to download, anything else to quit.")
        readline()[1] == 'y' || return
        download(vggurl,o[:model])
    end
    info("Reading $(o[:model])")
    vgg = matread(o[:model])
    params = get_params(vgg, atype)
    convnet = get_convnet(params...)
    description = vgg["meta"]["classes"]["description"]
    averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])
    info("Reading $(o[:image])")
    image = data(o[:image], averageImage)
    image = convert(atype, image)
    info("Classifying")
    @time y1 = convnet(image)
    z1 = vec(Array(y1))
    s1 = sortperm(z1,rev=true)
    @compat p1 = exp.(logp(z1))
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
    if VERSION >= v"0.5.0"
        c1 = permutedims(channelview(b1), (3,2,1))
    else
        c1 = separate(b1)
    end
    d1 = convert(Array{Float32}, c1)
    e1 = reshape(d1[:,:,1:3], (224,224,3,1))
    f1 = (255 * e1 .- averageImage)
    g1 = permutedims(f1, [2,1,3,4])
end

# This procedure makes pretrained MatConvNet VGG parameters convenient for Knet
# Also, if you want to extract features, specify the last layer you want to use
function get_params(CNN, atype; last_layer="prob")
    layers = CNN["layers"]
    weights, operations, derivatives = [], [], []

    for l in layers
        get_layer_type(x) = startswith(l["name"], x)
        operation = filter(x -> get_layer_type(x), LAYER_TYPES)[1]
        push!(operations, operation)
        push!(derivatives, haskey(l, "weights") && length(l["weights"]) != 0)

        if derivatives[end]
            w = l["weights"]
            if operation == "conv"
                w[2] = reshape(w[2], (1,1,length(w[2]),1))
            elseif operation == "fc"
                w[1] = transpose(mat(w[1]))
            end
            push!(weights, w)
        end

        last_layer != nothing && get_layer_type(last_layer) && break
    end

    map(w -> map(wi->convert(atype,wi), w), weights), operations, derivatives
end

# get convolutional network by interpreting parameters
function get_convnet(weights, operations, derivatives)
    function convnet(xs)
        i, j = 1, 1
        num_weights, num_operations = length(weights), length(operations)
        while i <= num_operations && j <= num_weights
            if derivatives[i]
                xs = forw(xs, operations[i], weights[j])
                j += 1
            else
                xs = forw(xs, operations[i])
            end

            i += 1
        end
        convert(Array{Float32}, xs)
    end
end

# convolutional network operations
convx(x,w) = conv4(w[1], x; padding=1, mode=1) .+ w[2]
if VERSION >= v"0.6.0"
    relux(x) = relu.(x)
else
    relux(x) = relu(x)
end
poolx = pool
probx(x) = x
fcx(x,w) = w[1] * mat(x) .+ w[2]
tofunc(op) = eval(parse(string(op, "x")))
forw(x,op) = tofunc(op)(x)
forw(x,op,w) = tofunc(op)(x,w)

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia vgg.jl cat.jpg
# julia> VGG.main("cat.jpg")
if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE=="vgg.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

end # module
