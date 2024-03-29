using PyCall, FileIO, ImageCore, ImageTransformations, CUDA, Tar, SHA
import Knet
using Knet.Layers21, Knet.Ops21
using Knet.Train20: param
using Knet.Ops20: softmax
typename(p::PyObject) = pytypeof(p).__name__
tf = pyimport("tensorflow")
include("../Models21.jl")


# Model translator
function keras2knet(p::PyObject; xtest = nothing, nlayers = -1)
    layers = nlayers >= 1 ? p.layers[1:nlayers] : copy(p.layers)
    model = Block(; name=p.name)
    while !isempty(layers)
        layer1 = popfirst!(layers)
        ltype = typename(layer1)
        # This could also pop one or more layers:
        layer2 = eval(Meta.parse("keras_$ltype"))(layer1, layers)
        layer2 !== nothing && push!(model, layer2)
    end
    if xtest !== nothing # x = rand(Float32,1,224,224,3)
        y1 = p.predict(xtest)
        y2 = model(xtest)
        @assert vec(Array(y1)) ≈ vec(Array(y2))
    end
    return model
end


keras_InputLayer(l, layers) = nothing


keras_ZeroPadding2D(l, layers) = ZeroPad(l.padding)


function keras_Conv2D(l, layers)
    # channels_last corresponds to inputs with shape nhwc (default)
    # channels_first corresponds to inputs with shape nchw
    # filters in channels_last are in hwxy order
    # knet default is whcn for images and whxy for filters
    w = param(permutedims(l.weights[1].numpy(),(2,1,3,4)))
    bias = (l.bias === nothing ? nothing : param(reshape(l.bias.numpy(), (1,1,:,1))))
    normalization, activation = keras_normact(layers)
    channelmajor = false
    crosscorrelation = true
    dilation = l.dilation_rate
    groups = l.groups
    padding = (l.padding == "valid" ? 0 : l.padding == "same" ? (size(w)[1:2] .- 1) .÷ 2 : l.padding)
    stride = l.strides
    Conv(w; activation, normalization, bias, channelmajor, crosscorrelation, dilation, groups, padding, stride)
end


function keras_DepthwiseConv2D(l, layers)
    w = l.weights[1].numpy()
    @assert size(w,4) == 1
    w = param(permutedims(w, (2,1,4,3))) ### (3,3,32,1) => (3,3,1,32)
    bias = (l.bias === nothing ? nothing : param(reshape(l.bias.numpy(), (1,1,:,1))))
    normalization, activation = keras_normact(layers)
    channelmajor = false
    crosscorrelation = true
    dilation = l.dilation_rate
    groups = size(w,4) ###
    padding = (l.padding == "valid" ? 0 : l.padding == "same" ? (size(w)[1:2] .- 1) .÷ 2 : l.padding)
    stride = l.strides
    Conv(w; activation, normalization, bias, channelmajor, crosscorrelation, dilation, groups, padding, stride)
end


function keras_normact(layers)
    if !isempty(layers) && typename(layers[1]) == "BatchNormalization"
        normalization = keras_BatchNormalization(popfirst!(layers), layers)
    else
        normalization = nothing
    end
    if !isempty(layers) && typename(layers[1]) == "ReLU"
        activation = keras_ReLU(popfirst!(layers), layers)
    else
        activation = nothing
    end
    normalization, activation
end    


function keras_ReLU(l, layers)
    max_value = (l.max_value === nothing || l.max_value[] === nothing ? Inf : l.max_value[])
    threshold = l.threshold[]
    negative_slope = l.negative_slope[]
    if threshold == negative_slope == 0
        Op(relu; max_value)
    else
        Op(relu; max_value, threshold, negative_slope)
    end
end


function keras_BatchNormalization(l, layers)
    bparam(x) = param(reshape(x.numpy(), (1,1,:,1)))
    mean = bparam(l.moving_mean).value
    var = bparam(l.moving_variance).value
    bias = bparam(l.beta)
    scale = bparam(l.gamma)
    epsilon = l.epsilon
    update = 1-l.momentum
    BatchNorm(; mean, var, bias, scale, epsilon, update)
end


function keras_GlobalAveragePooling2D(l, layers)
    x->begin #GlobalAveragePooling2D
        y = pool(x; mode=mean, window=typemax(Int))
        reshape2d(y)
    end
end


function keras_Reshape(l, layers)
    Op(reshape, (l.target_shape..., :))
end


function keras_Dropout(l, layers)
    Op(dropout, l.rate)
end


function keras_Activation(l, layers)
    if l.activation.__name__ == "softmax"
        Op(softmax; dims=1)
    elseif l.activation.__name__ == "swish"
        swish
    else
        error("Activation $(l.activation.__name__) not yet implemented.")
    end
end


function keras_Dense(l, layers)
    w = param(permutedims(l.weights[1].numpy()))
    bias = param(l.bias.numpy())
    activation = keras_Activation(l, layers)
    Linear(w; bias, activation)
end


function keras_Add(l, layers)
    @warn "Add not defined" maxlog=1
    Op("Add")
end

function keras_Rescaling(l, layers)
    @warn "Rescaling not defined" maxlog=1
    Op("Rescaling")
end

function keras_TFOpLambda(l, layers)
    @warn "TFOpLambda not defined" maxlog=1
    Op("TFOpLambda")
end


function keras_Multiply(l, layers)
    @warn "Multiply not defined" maxlog=1
    Op("Multiply")
end


function keras_Flatten(l, layers)
    @warn "Flatten not defined" maxlog=1
    Op("Flatten")
end


function keras_Normalization(l, layers)
    @warn "Normalization not defined" maxlog=1
    Op("Normalization")
end



### Test and import utils

function kerastest(img="ILSVRC2012_val_00000001.JPEG")
    global pf, jf, px, jx
    pf = tf.keras.applications.MobileNet()
    jf = keras2knet(pf)
    px = imagenet_preprocess(img; normalize=mobilenet_normalize_tf, format="nhwc", atype=Array{Float32}) # 1,224,224,3
    jx = Knet.atype(permutedims(px,(3,2,4,1))) # 224,224,3,1
    @time py = pf(px).numpy() |> vec
    @time jy = jf(jx) |> Array |> vec
    @show jy ≈ py
    idx = sortperm(jy, rev=true)
    [ idx jy[idx] imagenet_labels()[idx] ]
end


function tf_layer_outputs(model, img="ILSVRC2012_val_00000001.JPEG"; normalize=identity, resolution=224)
    model.trainable = false
    x = imagenet_preprocess(img; normalize, format="nhwc", atype=Array{Float32}, resolution) # 1,224,224,3
    outputnames = string.(l.output for l in model.layers)
    outputs = []
    for l in Knet.progress(model.layers)
        if l.input isa PyObject
            x = l(x)
        else
            xs = map(string.(l.input)) do s
                outputs[findfirst(isequal(s), outputnames)]
            end
            x = l(xs)
        end
        push!(outputs, x)
    end
    (x->x.numpy()).(outputs)
end


function tf_layer_outputs_slow(model, img="ILSVRC2012_val_00000001.JPEG"; normalize=identity, resolution=224)
    model.trainable = false
    input = imagenet_preprocess(img; normalize, format="nhwc", atype=Array{Float32}, resolution) # 1,224,224,3
    # layer_output = base_model.get_layer(layer_name).output
    models = []
    for l in Knet.progress(model.layers)
        push!(models, tf.keras.models.Model(model.input, outputs=[l.output]))
    end
    outputs = []
    for m in Knet.progress(models)
        try 
            push!(outputs, m.predict(input))
        catch e
            push!(outputs, nothing)
        end
    end
    return outputs
end


function tf_layer_output(model, layer=length(model.layers), img="ILSVRC2012_val_00000001.JPEG"; normalize=identity, resolution=224)
    model.trainable = false
    input = imagenet_preprocess(img; normalize, format="nhwc", atype=Array{Float32}, resolution) # 1,224,224,3
    m = tf.keras.models.Model(model.input, outputs=[model.layers[layer].output])
    m.predict(input)
end


function test_mobilenet_v2(file="mobilenet_v2_100_224_tf.jld2", img="ILSVRC2012_val_00000001.JPEG")
    pf = tf.keras.applications.MobileNetV2() # tf.keras.applications.mobilenet_v2.MobileNetV2()
    px = imagenet_preprocess(img; normalize=mobilenet_normalize_tf, format="nhwc", atype=Array{Float32}, resolution=224) # 1,224,224,3
    jf = MobileNet("mobilenet_v2_100_224_tf")
    jx = Knet.atype(permutedims(px,(3,2,4,1))) # 224,224,3,1
    jf(jx)
    setweights!(jf, file)
    isapprox(vec(pf(px).numpy()), softmax(vec(jf(jx))))
end


function import_mobilenet_tf(
    ; width=1, resolution=224, version=1,
    img="ILSVRC2012_val_00000001.JPEG",
)
    # global pf, jf0, jf, jx, jy, px, py, w
    @assert width in (1, 0.75, 0.5, 0.25)
    @assert resolution in (224, 192, 160, 128)
    alpha = width
    input_shape = (resolution, resolution, 3)

    if version == 1
        pf == tf.keras.applications.MobileNet(; alpha, input_shape)
        block=DWConv
        layout=mobilenet_v1_layout
        output=1024
    elseif version == 2
        pf == tf.keras.applications.MobileNetV2() # TODO: (; alpha, input_shape)
        block=MBConv
        layout=mobilenet_v2_layout
        output=1280
    end

    save_atype = Knet.array_type[]; Knet.array_type[] = Array{Float32}
    px = imagenet_preprocess(img; normalize=mobilenet_normalize_tf, format="nhwc", atype=Array{Float32}, resolution) # 1,224,224,3
    jx = Knet.atype(permutedims(px,(3,2,4,1))) # 224,224,3,1

    jf = MobileNet(; width, resolution, block, layout, output)
    jf(jx) # init weights
    jf0 = keras2knet(pf)
    w = getweights(jf0)
    if version == 1
        # MobileNetV1 ends with conv instead of linear
        w[end] = reshape(w[end],1000)
        w[end-1] = permutedims(reshape(w[end-1],(:,1000)))
    end        
    setweights!(jf, w)

    @time py = pf(px).numpy() |> vec
    @time jy = jf(jx) |> vec |> softmax
    @assert @show jy ≈ py       # TODO: this does not match python script, preprocessing?
    Knet.array_type[] = save_atype

    alpha100 = round(Int, 100*width)
    model = "mobilenet_v$(version)_$(alpha100)_$(resolution)"
    saveweights("$(model).jld2", jf) 
    run(`tar cf $(model).tar $(model).jld2`)
    sha1 = Tar.tree_hash("$(model).tar")
    run(`gzip $(model).tar`)
    sha2 = open("$(model).tar.gz") do f; bytes2hex(sha256(f)); end
    idx = sortperm(py, rev=true)
    pred = [ idx py[idx] imagenet_labels()[idx] ]
    display(pred[1:5,:]); println()
    @info "git-tree-sha1 = \"$sha1\""
    @info "sha256 = \"$sha2\""
end

function import_mobilenet_v2(
    ; width=1, resolution=224,
    tfmodel = tf.keras.applications.MobileNetV2,
    img="ILSVRC2012_val_00000001.JPEG",
)
    # global pf, jf0, jf, jx, jy, px, py, w
    save_atype = Knet.array_type[]; Knet.array_type[] = Array{Float32}
    @assert width in (1, 0.75, 0.5, 0.25)
    @assert resolution in (224, 192, 160, 128)
    px = imagenet_preprocess(img; normalize=mobilenet_normalize_tf, format="nhwc", atype=Array{Float32}, resolution) # 1,224,224,3
    jx = Knet.atype(permutedims(px,(3,2,4,1))) # 224,224,3,1
    pf = tfmodel()  # ; alpha=width, input_shape=(resolution,resolution,3))
    jf0 = keras2knet(pf)
    w = getweights(jf0)
    jf = MobileNet(; width, resolution, block=MBConv, layout=mobilenet_v2_layout, output=1280)
    jf(jx) # init weights
    setweights!(jf, w)
    @time py = pf(px).numpy() |> vec
    @time jy = jf(jx) |> vec |> softmax
    @assert @show jy ≈ py
    alpha = round(Int, 100*width)
    model = "mobilenet_v2_$(alpha)_$(resolution)"
    saveweights("$(model).jld2", jf) 
    run(`tar cf $(model).tar $(model).jld2`)
    sha1 = Tar.tree_hash("$(model).tar")
    run(`gzip $(model).tar`)
    sha2 = open("$(model).tar.gz") do f; bytes2hex(sha256(f)); end
    idx = sortperm(py, rev=true)
    @info "git-tree-sha1 = \"$sha1\""
    @info "sha256 = \"$sha2\""
    Knet.array_type[] = save_atype
    [ idx py[idx] imagenet_labels()[idx] ]
end


function import_mobilenet_v3(   # WIP
    ; 
    tfmodel = tf.keras.applications.MobileNetV3Small,
    layout = mobilenet_v3_small_layout,
    img="ILSVRC2012_val_00000001.JPEG",
    width=1,
    resolution=224)
    global pf, jf0, jf, jx, jy, px, py, w0
    save_atype = Knet.array_type[]; Knet.array_type[] = Array{Float32}
    @assert width in (1, 0.75, 0.5, 0.25)
    @assert resolution in (224, 192, 160, 128)
    # px = imagenet_preprocess(img; normalization="tf", format="nhwc", atype=Array{Float32}, resolution) # 1,224,224,3
    px = imagenet_preprocess(img; format="nhwc", atype=Array{Float32}, resolution) # 1,224,224,3
    px = 255 .* px #   x = layers.Rescaling(scale=1. / 127.5, offset=-1.)(x)
    # jx = Knet.atype(permutedims(px,(3,2,4,1))) # 224,224,3,1
    jx = imagenet_preprocess(img; format="whcn", atype=Array{Float32}, resolution) # 224,224,3,1
    jx = 2 .* jx .- 1
    pf = tfmodel()  # ; alpha=width, input_shape=(resolution,resolution,3))
    jf0 = keras2knet(pf)
    w0 = getweights(jf0)
    # MobileNet_tf ends with conv instead of linear
    w0[end-3] = permutedims(reshape2d(w0[end-3]))
    w0[end-2] = vec(w0[end-2])
    w0[end-1] = permutedims(reshape2d(w0[end-1]))
    w0[end] = vec(w0[end])
    jf = MobileNet(; width=1, resolution=224, input=16, output=(576,1024), classes=1000, activation=hardswish, tfpadding=true, bnupdate=0.001, bnepsilon=0.001, dropout=0.2, block=mobilenet_v3_block, layout)
    jf(jx) # init weights
    setweights!(jf, w0)
    @time py = pf(px).numpy() |> vec
    @time jy = jf(jx) |> vec |> softmax
    @show jy ≈ py
    # alpha = round(Int, 100*width)
    # model = "mobilenet_v2_$(alpha)_$(resolution)"
    # saveweights("$(model).jld2", jf) 
    # run(`tar cf $(model).tar $(model).jld2`)
    # sha1 = Tar.tree_hash("$(model).tar")
    # run(`gzip $(model).tar`)
    # sha2 = open("$(model).tar.gz") do f; bytes2hex(sha256(f)); end
    # @info "git-tree-sha1 = \"$sha1\""
    # @info "sha256 = \"$sha2\""
    pidx = sortperm(py, rev=true)
    jidx = sortperm(jy, rev=true)
    Knet.array_type[] = save_atype
    [ pidx py[pidx] imagenet_labels()[pidx] jidx jy[jidx] imagenet_labels()[jidx] ]
end


function import_efficientnet(n=0)
    # to test with outs = tf_layer_outputs(pf; normalize=(x->(255 .* x)), resolution=240)
    tfmodel = getproperty(tf.keras.applications, "EfficientNetB$n")
    img="ILSVRC2012_val_00000001.JPEG"
    global pf, jf0, jf, jx, jy, px, py, w0
    model = lowercase(tfmodel.__name__)
    opts = efficientnet_models[model]
    save_atype = Knet.array_type[]; Knet.array_type[] = Array{Float32}
    px = imagenet_preprocess(img; normalize=(x->(255*x)), format="nhwc", atype=Array{Float32}, opts.resolution) # 1,224,224,3
    jx = imagenet_preprocess(img; normalize=efficientnet_normalize, format="whcn", atype=Array{Float32}, opts.resolution) # 224,224,3,1
    pf = tfmodel()  # ; alpha=width, input_shape=(resolution,resolution,3))
    jf0 = keras2knet(pf)
    w0 = getweights(jf0)
    jf = EfficientNet(; opts...)
    jf(jx) # init weights
    setweights!(jf, w0)
    @info "Calling tf model"
    @time py = pf(px).numpy() |> vec
    @info "Calling julia model"
    @time jy = jf(jx) |> vec |> softmax
    @show jy ≈ py
    saveweights("$(model).jld2", jf) 
    run(`tar cf $(model).tar $(model).jld2`)
    sha1 = Tar.tree_hash("$(model).tar")
    run(`gzip $(model).tar`)
    sha2 = open("$(model).tar.gz") do f; bytes2hex(sha256(f)); end
    @info "git-tree-sha1 = \"$sha1\""
    @info "sha256 = \"$sha2\""
    pidx = sortperm(py, rev=true)
    jidx = sortperm(jy, rev=true)
    Knet.array_type[] = save_atype
    [ pidx py[pidx] imagenet_labels()[pidx] jidx jy[jidx] imagenet_labels()[jidx] ]
end


