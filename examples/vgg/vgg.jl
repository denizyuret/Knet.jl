using Pkg; for p in ("Knet","ArgParse"); haskey(Pkg.installed(),p) || Pkg.add(p); end

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
using Knet,ArgParse
include(Knet.dir("data","imagenet.jl"))

const imgurl = "https://github.com/BVLC/caffe/raw/master/examples/images/cat.jpg"
const vggurl = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat"
const LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]

function main(args=ARGS)
    s = ArgParseSettings()
    s.description="vgg.jl (c) Deniz Yuret, İlker Kesen, 2016. Classifying images with the VGG model from http://www.robots.ox.ac.uk/~vgg/research/very_deep."
    # s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("image"; default=imgurl; help="Image file or URL.")
        ("--model"; default="imagenet-vgg-verydeep-16"; help="Model name")
        ("--top"; default=5; arg_type=Int; help="Display the top N classes")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array and float type to use")
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    println(s.description)
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    atype = eval(Meta.parse(o[:atype]))

    global _vggcache
    if !@isdefined(_vggcache); _vggcache=Dict(); end
    if !haskey(_vggcache,o[:model])
        vgg = matconvnet(o[:model])
        params = get_params(vgg, atype)
        convnet = get_convnet(params...)
        description = vgg["meta"]["classes"]["description"]
        averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])
        _vggcache[o[:model]] = vgg, params, convnet, description, averageImage
    else
        vgg, params, convnet, description, averageImage = _vggcache[o[:model]]
    end

    image = imgdata(o[:image], averageImage)
    image = convert(atype, image)
    @info("Classifying")
    @time y1 = convnet(image)
    z1 = vec(Array(y1))
    s1 = sortperm(z1,rev=true)
    p1 = exp.(logp(z1))
    display(hcat(p1[s1[1:o[:top]]], description[s1[1:o[:top]]]))
    println()
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
            w = copy(l["weights"])
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
relux(x) = relu.(x)
poolx = pool
probx(x) = x
fcx(x,w) = w[1] * mat(x) .+ w[2]
tofunc(op) = eval(Meta.parse(string(op, "x")))
forw(x,op) = tofunc(op)(x)
forw(x,op,w) = tofunc(op)(x,w)

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia vgg.jl cat.jpg
# julia> VGG.main("cat.jpg")
PROGRAM_FILE=="vgg.jl" && main(ARGS)

end # module
