#model url: https://github.com/FluxML/Metalhead.jl/releases/download/v0.1.1/resnet.bson
#reference: https://github.com/FluxML/Metalhead.jl
using Knet, KnetLayers, BSON, Images

struct ResidualBlock
  layers
  shortcut
end

function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, shortcut = identity)
  layers = []
  for i in 2:length(filters)
    push!(layers, Conv(activation=nothing, height=kernels[i-1][1], width=kernels[i-1][2], inout=filters[i-1]=>filters[i], padding = pads[i-1], stride = strides[i-1], mode=1, binit=nothing))
    if i != length(filters)
      push!(layers, Chain(BatchNorm(filters[i]),ReLU())) # I think we need batchnorm with relu activation
    else
      push!(layers, BatchNorm(filters[i]))
    end
  end
  ResidualBlock(Chain(layers...), shortcut)
end

ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity) =
  ResidualBlock(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], shortcut)

(r::ResidualBlock)(input) = relu.(r.layers(input) + r.shortcut(input))

function BasicBlock(filters::Int, downsample::Bool = false, res_top::Bool = false)
  # NOTE: res_top is set to true if this is the first residual connection of the architecture
  # If the number of channels is to be halved set the downsample argument to true
  if !downsample || res_top
    return ResidualBlock([filters for i in 1:3], [3,3], [1,1], [1,1])
  end
  shortcut = Chain(Conv(activation=nothing, height=3, width=3, inout=filters÷2=>filters, padding = (1,1), stride = (2,2), mode=1, binit=nothing), BatchNorm(filters))
  ResidualBlock([filters÷2, filters, filters], [3,3], [1,1], [1,2], shortcut)
end

function Bottleneck(filters::Int, downsample::Bool = false, res_top::Bool = false)
  # NOTE: res_top is set to true if this is the first residual connection of the architecture
  # If the number of channels is to be halved set the downsample argument to true
  if !downsample && !res_top
    ResidualBlock([4 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1])
  elseif downsample && res_top
    ResidualBlock([filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1], Chain(Conv(activation=nothing, height=1,width=1, inout=filters=>4 * filters, padding = (0,0), stride = (1,1), mode=1, binit=nothing), BatchNorm(4 * filters)))
  else
    shortcut = Chain(Conv(activation=nothing, height=1, width=1, inout=2 * filters=>4 * filters, padding = (0,0), stride = (2,2), mode=1, binit=nothing), BatchNorm(4 * filters))
    ResidualBlock([2 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,2], shortcut)
  end
end

const KA = KnetLayers.arrtype
transfer!(p::Param, x)  = copyto!(p.value,x)
transfer!(p, x) = copyto!(p,x)
to4D(x) = reshape(x,1,1,length(x),1)

function trained_resnet50_layers(model=KnetLayers.dir("examples","resnet.bson"))
    if !isfile(model)
        download("https://github.com/FluxML/Metalhead.jl/releases/download/v0.1.1/resnet.bson",model)
    end
    weight = BSON.load(model)
    weights = Dict{Any ,Any}()
    for ele in keys(weight)
        weights[string(ele)] = weight[ele]
    end
    ls = load_resnet(resnet_configs["resnet50"]...)
    transfer!(ls[1][1].weight, weights["gpu_0/conv1_w_0"])
    ls[1][2].moments.var =  KA(weights["gpu_0/res_conv1_bn_riv_0"] |> to4D)
    ls[1][2].moments.mean = KA(weights["gpu_0/res_conv1_bn_rm_0"] |> to4D)
    ls[1][2].params = KA(vcat(weights["gpu_0/res_conv1_bn_s_0"],weights["gpu_0/res_conv1_bn_b_0"]))
    count = 2
    for j in [3:5, 6:9, 10:15, 16:18]
        for p in j
            transfer!(ls[p].layers[1].weight, weights["gpu_0/res$(count)_$(p-j[1])_branch2a_w_0"] )
            ls[p].layers[2][1].moments.var = KA(weights["gpu_0/res$(count)_$(p-j[1])_branch2a_bn_riv_0"] |> to4D)
            ls[p].layers[2][1].moments.mean = KA(weights["gpu_0/res$(count)_$(p-j[1])_branch2a_bn_rm_0"] |> to4D)
            ls[p].layers[2][1].params =  KA(vcat(weights["gpu_0/res$(count)_$(p-j[1])_branch2a_bn_s_0"],weights["gpu_0/res$(count)_$(p-j[1])_branch2a_bn_b_0"]))
            transfer!(ls[p].layers[3].weight , weights["gpu_0/res$(count)_$(p-j[1])_branch2b_w_0"])
            ls[p].layers[4][1].moments.var = KA(weights["gpu_0/res$(count)_$(p-j[1])_branch2b_bn_riv_0"] |> to4D)
            ls[p].layers[4][1].moments.mean = KA(weights["gpu_0/res$(count)_$(p-j[1])_branch2b_bn_rm_0"] |> to4D)
            ls[p].layers[4][1].params =  KA(vcat(weights["gpu_0/res$(count)_$(p-j[1])_branch2b_bn_s_0"],weights["gpu_0/res$(count)_$(p-j[1])_branch2b_bn_b_0"]))
            transfer!(ls[p].layers[5].weight , weights["gpu_0/res$(count)_$(p-j[1])_branch2c_w_0"])
            ls[p].layers[6].moments.var = KA(weights["gpu_0/res$(count)_$(p-j[1])_branch2c_bn_riv_0"] |> to4D)
            ls[p].layers[6].moments.mean = KA(weights["gpu_0/res$(count)_$(p-j[1])_branch2c_bn_rm_0"] |> to4D)
            ls[p].layers[6].params =  KA(vcat(weights["gpu_0/res$(count)_$(p-j[1])_branch2c_bn_s_0"],weights["gpu_0/res$(count)_$(p-j[1])_branch2c_bn_b_0"]))
        end
        transfer!(ls[j[1]].shortcut[1].weight , weights["gpu_0/res$(count)_0_branch1_w_0"])
        ls[j[1]].shortcut[2].moments.var = KA(weights["gpu_0/res$(count)_0_branch1_bn_riv_0"] |> to4D)
        ls[j[1]].shortcut[2].moments.mean = KA(weights["gpu_0/res$(count)_0_branch1_bn_rm_0"] |> to4D)
        ls[j[1]].shortcut[2].params =  KA(vcat(weights["gpu_0/res$(count)_0_branch1_bn_s_0"], weights["gpu_0/res$(count)_0_branch1_bn_b_0"]))
        count += 1
    end
    transfer!(ls[21].mult.weight, permutedims(weights["gpu_0/pred_w_0"], (2,1)));
    transfer!(ls[21].bias, weights["gpu_0/pred_b_0"]);
    return ls
end

function load_resnet(Block, layers, initial_filters::Int = 64, nclasses::Int = 1000)
  local top = []
  local residual = []
  local bottom = []

  push!(top, Chain(Conv(activation=nothing, width=7, height=7, inout=3=>initial_filters, padding = (3,3), stride = (2,2), mode=1, binit=nothing), BatchNorm(initial_filters)))
  push!(top, Pool(window=(3,3), padding = (1,1), stride = (2,2)))

  for i in 1:length(layers)
    push!(residual, Block(initial_filters, true, i==1))
    for j in 2:layers[i]
      push!(residual, Block(initial_filters))
    end
    initial_filters *= 2
  end

  push!(bottom, Pool(window=(7,7), mode=1))
  push!(bottom, x -> mat(x))
  if Block == Bottleneck
    push!(bottom, (Linear(input=2048,output=nclasses)))
  else
    push!(bottom, (Dense(input=512,output=nclasses)))
  end
  push!(bottom, softmax)

  Chain(top..., residual..., bottom...)
end

resnet_configs =
  Dict("resnet18" => (BasicBlock, [2, 2, 2, 2]),
       "resnet34" => (BasicBlock, [3, 4, 6, 3]),
       "resnet50" => (Bottleneck, [3, 4, 6, 3]),
       "resnet101" => (Bottleneck, [3, 4, 23, 3]),
       "resnet152" => (Bottleneck, [3, 8, 36, 3]))

struct ResNet18
  layers::Chain
end

ResNet18() = ResNet18(load_resnet(resnet_configs["resnet18"]...))

trained(::Type{ResNet18}) = error("Pretrained Weights for ResNet18 are not available")

Base.show(io::IO, ::ResNet18) = print(io, "ResNet18()")

(m::ResNet18)(x) = m.layers(x)

struct ResNet34
  layers::Chain
end

ResNet34() = ResNet34(load_resnet(resnet_configs["resnet34"]...))

trained(::Type{ResNet34}) = error("Pretrained Weights for ResNet34 are not available")

Base.show(io::IO, ::ResNet34) = print(io, "ResNet34()")

(m::ResNet34)(x) = m.layers(x)

struct ResNet50
  layers::Chain
end

ResNet50() = ResNet50(load_resnet(resnet_configs["resnet50"]...))

trained(::Type{ResNet50}) = ResNet50(trained_resnet50_layers())

Base.show(io::IO, ::ResNet50) = print(io, "ResNet50()")

(m::ResNet50)(x) = m.layers(x)

struct ResNet101
  layers::Chain
end

ResNet101() = ResNet101(load_resnet(resnet_configs["resnet101"]...))

trained(::Type{ResNet101}) = error("Pretrained Weights for ResNet101 are not available")

Base.show(io::IO, ::ResNet101) = print(io, "ResNet101()")

(m::ResNet101)(x) = m.layers(x)

struct ResNet152
  layers::Chain
end

ResNet152() = ResNet152(load_resnet(resnet_configs["resnet152"]...))

trained(::Type{ResNet152}) = error("Pretrained Weights for ResNet152 are not available")

Base.show(io::IO, ::ResNet152) = print(io, "ResNet152()")

(m::ResNet152)(x) = m.layers(x)

function preprocess(img::AbstractMatrix{<:AbstractRGB})
  # Resize such that smallest edge is 256 pixels long
    img = resize_smallest_dimension(img, 256)
    im = center_crop(img, 224)
    z = (channelview(im) .* 255 .- 128)./128;
    z1 = Float32.(permutedims(z, (3, 2, 1))[:,:,:,:]);
end

# Resize an image such that its smallest dimension is the given length
function resize_smallest_dimension(im, len)
  reduction_factor = len/minimum(size(im)[1:2])
  new_size = size(im)
  new_size = (
      round(Int, size(im,1)*reduction_factor),
      round(Int, size(im,2)*reduction_factor),
  )
  if reduction_factor < 1.0
    # Images.jl's imresize() needs to first lowpass the image, it won't do it for us
    im = imfilter(im, KernelFactors.gaussian(0.75/reduction_factor), Inner())
  end
  return imresize(im, new_size)
end

# Take the len-by-len square of pixels at the center of image `im`
function center_crop(im, len)
  l2 = div(len,2)
  adjust = len % 2 == 0 ? 1 : 0
  return im[div(end,2)-l2:div(end,2)+l2-adjust,div(end,2)-l2:div(end,2)+l2-adjust]
end
