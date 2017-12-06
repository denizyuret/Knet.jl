for p in ("Knet",)
    (Pkg.installed(p) == nothing) && Pkg.add(p)
end
include(Pkg.dir("Knet","data","imagenet.jl"))

# TODO: improve example and document metadata return type further
# TODO: document low-level API
"""This module implements the ResNet 50,101,150 and CIFAR models from
'Deep Residual Learning for Image Regocnition', Kaiming He, Xiangyu Zhang,
Shaoqing Ren, Jian Sun, arXiv technical report 1512.03385, 2015.

* Paper url: https://arxiv.org/abs/1512.03385
* Project page: https://github.com/KaimingHe/deep-residual-networks
* MatConvNet weight: http://www.vlfeat.org/matconvnet/pretrained

-------------------------------
Initialization functions of ImageNet models:

 `w, m, meta = resnet50init( ;kwargs...)`

 `w, m, meta = resnet101init(;kwargs...)`
 
 `w, m, meta = resnet152init(;kwargs...)`

# Return value
All initialization functions return 3-element tuples of the form `(weights, moments, metadata)`.
The `metadata` is pretrained model's "meta" cell if `trained=true`, `nothing` otherwise.
`moments` are to be used in `batchnorm` operations.

# Keyword arguments
 `trained=true`: if true, corresponding pretrained model is loaded. Otherwise, 
models are initialized from scratch.
          
 `etype=Float32`: numeric type of the parameters and moments

 `stage=0`: Used when an earlier stage of the input used as feature. `stage=0` means full
classification is desired. `stage=n` means the output after the nth residual layer group
is desided, when 1<=n<=4. For n=5, global average pooling is also included. 
An error is thrown for n > 5.  

-------------------------------

Forward functions of ImageNet models:

 `y = resnet50(w, m, x; stage)`

 `y = resnet101(w, m, x; stage)`

 `y = resnet152(w, m, x; stage)`

# Arguments
    `w`, `m` and `x` correspond to weights, batchnorm moments and input.

# Keywords
    `stage=0`: Same argument provided to the init functions.

# Return Values
    functions returns an array of size (1000, batch_size).


# Usage Example for Classification
    w, m, meta = resnet101init()
    # use meta to resize, crop and normalize x
    y = resnet101(w, m, x)

---------------------------
CIFAR models can be used as:

   `w, m = resnetcifarinit(110; nclasses=10) #nclasses=100 for CIFAR100`
    
   `y = resnetcifar(w, m, x)`
  
"""
module ResNetLib

using Knet

resnet50init(;  o...)  = resnetinit([3, 4, 6,  3]; o...)
resnet101init(; o...)  = resnetinit([3, 4, 23, 3]; o...)
resnet152init(; o...)  = resnetinit([3, 8, 36, 3]; o...)

resnet50( ws, ms, x; o...)  = resnet(ws, ms, x, [3, 4, 6,  3]; o...)
resnet101(ws, ms, x; o...)  = resnet(ws, ms, x, [3, 4, 23, 3]; o...)
resnet152(ws, ms, x; o...)  = resnet(ws, ms, x, [3, 8, 36, 3]; o...)

resnetcifarinit(depth::Int; nclasses=10) =
    let n = div(depth-2, 3)
        dataset = nclasses == 10 ? :cifar10 : :cifar100
        nrep = div(n, 2)
        # call the generic fn
        resnetinit([nrep, nrep, nrep];
                   dataset=dataset,
                   channels=[16,32,64],
                   resize=[false,true,true],
                   trained=false,
                   blockinit=basicinit)
    end

resnetcifar(w, m, x) =
    let depth = div(length(w)-4, 2)
        n = div(depth-2, 3)
        nrep = div(n, 2)
        nclasses = length(w[end])
        dataset = nclasses == 10 ? :cifar10 : :cifar100
        # call the generic fn
        resnet(w, m, x, [nrep, nrep, nrep];
               strides=[1,2,2],
               resize=[false,true,true],
               block=basic,
               dataset=dataset)
    end


const basic_channels = [64, 128, 256, 512]
const bneck_channels = 4basic_channels

function resnetinit(repeats;
                    channels=bneck_channels,
                    resize=[true, true, true, true], #size changes require 1x1 conv shortcuts
                    etype=Float32,
                    atype=(gpu() >= 0) ? KnetArray : Array,
                    trained=true,
                    blockinit=bneckinit,
                    stage=0,
                    dataset=:imagenet)
    @assert (dataset in [:imagenet, :cifar10, :cifar100]) "dataset kwarg should be :imagenet, :cifar10, or :cifar100"
    @assert (stage <= length(repeats)+1) "Use stage=0 to perform classification"
    ws = []
    ms = []
    if dataset === :imagenet
        push!(ws, cinit([7,7,3,64], etype))
        push!(ws, bnparams(etype, 64))
        channels = [64, channels...]
    else
        push!(ws, cinit([3,3,3,channels[1]], etype))
        push!(ws, bnparams(etype, channels[1]))
        channels = [channels[1], channels...]
    end
    push!(ms, bnmoments())
    for i = 1:length(repeats)
        w, m = layerinit(repeats[i], blockinit,
                         channels[i], channels[i+1];
                         downsample=resize[i],
                         etype=etype)
        push!(ws, w...)
        push!(ms, m...)
        (stage == i) && break
    end
    if stage == 0 
        nclasses = dataset == :cifar10 ? 10 : dataset == :cifar100 ? 100 : 1000
        push!(ws, xavier(etype, nclasses, channels[end]))
        push!(ws, zeros(etype, nclasses, 1))
    end
    if trained
        depth = sum(repeats .* (blockinit == bneckinit ? 3 : 2)) + 2
        metadata = load_resnet!(ws, ms; atype=atype, stage=stage, depth=depth)
    else
        metadata = nothing
    end
    ws = map(atype, ws)
    return ws, ms, metadata #todo: add non-matconvnet metadata
end


function resnet(ws, ms, x, repeats;
                block=bneck,
                resize=[true, true, true, true],
                strides=[1,2,2,2],
                stage=0,
                dataset=:imagenet)
    @assert (dataset in [:imagenet, :cifar10, :cifar100]) "dataset kwarg should be :imagenet, :cifar10, or :cifar100"

    if dataset === :imagenet
        o = conv4(ws[1], x; padding=3, stride=2)
        # There is a forgotten bias in resnet50 pre-trained model
        # which happens to learn some small values.
        # This hack is to support that.
        if ndims(ws[end]) == 4
            o = o .+ last(ws)
            ws = ws[1:end-1]
        end
        o = batchnorm(o, ms[1], ws[2])
        o = relu.(o)
        o = pool(o; window=3, stride=2)
    else
        o = conv4(ws[1], x; padding=1)
        o = batchnorm(o, ms[1], ws[2])
        o = relu.(o)
    end
    wstart = 3
    mstart = 2
    for (i, r) in enumerate(repeats)
        d = resize[i]
        wi, mi = nlparams(r, d, block)
        o = layer(ws[wstart:wstart+wi-1],
                  ms[mstart:mstart+mi-1],
                  o, r, block;
                  stride=strides[i],
                  downsample=d)
        wstart += wi
        mstart += mi
        (stage == i) && return o
    end
    # global average pooling & output
    o = pool(o; window=size(o,1,2), mode=2)
    stage == length(repeats)+1 && return o
    return ws[end-1] * mat(o) .+ ws[end]
end


nlparams(repeat, downsample, block) =
    let np = (block == bneck) ? 6 : 4
        nm = div(np, 2)
        nd = Int(downsample)
        repeat * np + 2nd, repeat * nm + nd
    end


function layerinit(repeat, blockinit, input, output;
                   downsample=true, etype=Float32)
    @assert (blockinit in [basicinit, bneckinit]) "Unknown blockinit function"
    ws, ms = [], []
    # add downsample
    for i = 1:repeat
        w, m = blockinit(input, output;
                         downsample=downsample, etype=etype)
        push!(ws, w...); push!(ms, m...)
        input = output
        downsample = false
    end
    return ws, ms
end

function layer(w, m, x, repeat, block;
               stride=1, downsample=true)
    @assert (block in [basic, bneck]) "Unknown block function"
    nparams = (block == bneck) ? 6 : 4
    nmoments = div(nparams, 2)
    wstart = 1
    mstart = 1
    for i = 1:repeat
        winc = nparams + (downsample?2:0)
        minc = nmoments + (downsample?1:0)
        # run the block
        x = block(w[wstart:wstart+winc-1], m[mstart:mstart+minc-1], x;
                  downsample=downsample, stride=stride)
        # update metadata
        wstart += winc
        mstart += minc
        downsample = false
        stride = 1
    end
    return x
end

# BOTTLENECK BLOCK
function bneckinit(input, output; downsample=false, etype=Float32)
    planes = div(output, 4)
    w = Any[
        (downsample ? [cinit([1, 1, input, output], etype), bnparams(etype,output)] : [])...,
        cinit([1, 1, input,  planes],  etype),  bnparams(etype, planes),
        cinit([3, 3, planes, planes],  etype),  bnparams(etype, planes),
        cinit([1, 1, planes, output],  etype),  bnparams(etype, output)
    ]
    m = Any[bnmoments() for i=1:div(length(w), 2)]
    return w, m
end


function bneck(w, m, x; stride=1, downsample=false)
    if downsample
        x_ = batchnorm(conv4(w[1], x; stride=stride), m[1], w[2])
        w = w[3:end]
        m = m[2:end]
    else
        x_ = x
    end
    o = relu.(batchnorm(conv4(w[1], x; stride=stride), m[1], w[2]))
    o = relu.(batchnorm(conv4(w[3], o; padding=1), m[2], w[4]))
    o = batchnorm(conv4(w[5], o), m[3], w[6])
    return relu.(x_ .+ o)
end

# BASIC BLOCK
function basicinit(input, output; downsample=false, etype=Float32)
    w = Any[
        (downsample ? [cinit([1, 1, input, output], etype), bnparams(etype,output)] : [])...,
        cinit([3, 3, input,  output],  etype),  bnparams(etype, output),
        cinit([3, 3, output, output],  etype),  bnparams(etype, output)
    ]
    m = Any[bnmoments() for i=1:div(length(w), 2)]
    return w, m
end

function basic(w, m, x; stride=1, downsample=false)
    if downsample
        x_ = batchnorm(conv4(w[1], x; stride=stride), m[1], w[2])
        w = w[3:end]
        m = m[2:end]
    else
        x_ = x
    end
    o = relu.(batchnorm(conv4(w[1], x; padding=1, stride=stride), m[1], w[2]))
    o = relu.(batchnorm(conv4(w[3], o; padding=1), m[2], w[4]))
    return relu.(x_ .+ o)
end



cinit(dims, etype=Float32) = etype(sqrt(2 / prod(dims[[1,2,4]]))) .* randn(etype, dims...)


function load_resnet!(weights, moments;
                      modeldir=Knet.dir("data"),
                      stage=0, depth=101,
                      o...)
    info("Loading pretrained weights...")
    # All of this is implemented in Knet/data/imagenet.jl
    # urls = Dict([
    #     50  => "http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat",
    #     101 => "http://www.vlfeat.org/matconvnet/models/imagenet-resnet-101-dag.mat",
    #     152 => "http://www.vlfeat.org/matconvnet/models/imagenet-resnet-152-dag.mat",
    # ])
    # filename = "imagenet-resnet-$depth-dag.mat"
    # dest = joinpath(modeldir, filename)
    # if ~isfile(dest)
    #     info("Downloading resnet ", depth, " weights to ", dest)
    #     download(urls[depth], dest)
    # end
    # r = matread(dest)
    r = Main.matconvnet("imagenet-resnet-$depth-dag")
    load_params!(weights, moments, r["params"];
                 first_bias=depth==50,
                 stage=stage,
                 o...)
    return r["meta"]
end

function load_params!(weights, moments, matparams;
                      first_bias=false,
                      atype=(gpu()>=0 ? KnetArray : Array),
                      stage=0)
    et = eltype(weights[1])
    params = matparams["value"]
    params = map(x->et.(x), params)
    if first_bias
        b = params[2]
        params = Any[params[1], params[3:end]...]
        push!(weights, reshape(atype(b), (1,1,length(b),1)))
    end
    len = length(params)
    wc = 1
    mc = 1
    for i = 1:4:len-2
        copy!(weights[wc], params[i])
        bnw = weights[wc+1]
        copy!(bnw, vcat(params[i+1][:], params[i+2][:]))
        m = params[i+3]
        sz = (1, 1, size(m, 1), 1)
        moments[mc].mean = reshape(atype(m[:, 1]), sz) 
        moments[mc].var  = reshape(atype(m[:, 2]), sz)
        wc += 2
        mc += 1
        stage !== 0 && wc > length(weights) && return
    end
    (stage == 5) && return
    w, b = params[end-1], params[end]
    copy!(weights[wc], mat(w)')
    copy!(weights[wc+1], b)
end

end #module
