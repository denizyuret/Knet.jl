include("resnet-util.jl")

# need relu6(x) = clamp.(relu(x), 0, 6)

torch2knet(p::PyObject) = eval(Meta.parse(p._get_name()))(p)


function MobileNetV2(p::PyObject)
    s = torch2knet(p.features)
    push!(s, MobileNetOutput(p.classifier))
end


function MobileNetOutput(p::PyObject)
    Sequential(
        x->pool(x; mode=1, window=size(x)[1:2]),
        x->reshape(x, :, size(x,4)),
        Linear(param(t2a(p[2].weight)); bias=param(t2a(p[2].bias)), dropout=p[1].p),
    )
end


function Sequential(p::PyObject)
    s = Sequential()
    for l in p; push!(s, torch2knet(l)); end
    return s
end


function ConvBNReLU(p::PyObject)
    conv = torch2knet(p[1])
    conv.normalization = torch2knet(p[2])
    conv.activation = torch2knet(p[3])
    return conv
end


function Conv2d(p::PyObject; normalization=nothing, activation=nothing)
    w = param(permutedims(t2a(p.weight), (4,3,2,1)))
    bias = (p.bias === nothing ? nothing :
            param(reshape(t2a(p.bias), (1,1,:,1))))
    Conv(w; bias, normalization, activation,
         p.padding, p.stride, p.dilation, p.groups)
end


function BatchNorm2d(b::PyObject)
    b.track_running_stats = false # so we can do an inference comparison
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


function ReLU6(p::PyObject)
    return relu6
end

relu6(x)=clamp(x,0,6)

function InvertedResidual(p::PyObject)
    c = torch2knet(p.conv)
    if c[end] isa BatchNorm && c[end-1] isa Conv && c[end-1].normalization == nothing
        c[end-1].normalization = pop!(c)
    end
    all(c[2].stride .== 1) && size(c[1].w, 3) == size(c[end].w, 4) ? Residual(c, identity) : c
end

mv2 = models.mobilenet_v2(pretrained=true).eval()

a = torch2knet(mv2)


#=
channels: 16,32

julia> a
Sequential

Input:
224x3=>112x32
  Conv(3×3, 3=>32, padding=(1, 1), stride=(2, 2), BatchNorm(), relu6)

[t,c,n,s] = [exp factor, output channels, layers in sequence with same output, first layer stride]

16Channels:
[1,16,1,1]
112x32=>112x16
  Sequential
    Conv(3×3, 1=>32, padding=(1, 1), groups=32, BatchNorm(), relu6)
    Conv(1×1, 32=>16, BatchNorm())

24Channels:
[6, 24, 2, 2]-1
112x16=>56x24
  Sequential
    Conv(1×1, 16=>96, BatchNorm(), relu6)
    Conv(3×3, 1=>96, padding=(1, 1), stride=(2, 2), groups=96, BatchNorm(), relu6)
    Conv(1×1, 96=>24, BatchNorm())

[6, 24, 2, 2]-2
56x24=>56x24
  Residual
    Sequential
      Conv(1×1, 24=>144, BatchNorm(), relu6)
      Conv(3×3, 1=>144, padding=(1, 1), groups=144, BatchNorm(), relu6)
      Conv(1×1, 144=>24, BatchNorm())
    +
    identity

32Channels:
[6, 32, 3, 2]-1
56x24=>28x32
  Sequential
    Conv(1×1, 24=>144, BatchNorm(), relu6)
    Conv(3×3, 1=>144, padding=(1, 1), stride=(2, 2), groups=144, BatchNorm(), relu6)
    Conv(1×1, 144=>32, BatchNorm())

[6, 32, 3, 2]-2
28x32=>28x32-1
  Residual
    Sequential
      Conv(1×1, 32=>192, BatchNorm(), relu6)
      Conv(3×3, 1=>192, padding=(1, 1), groups=192, BatchNorm(), relu6)
      Conv(1×1, 192=>32, BatchNorm())
    +
    identity

[6, 32, 3, 2]-3
28x32=>28x32-2
  Residual
    Sequential
      Conv(1×1, 32=>192, BatchNorm(), relu6)
      Conv(3×3, 1=>192, padding=(1, 1), groups=192, BatchNorm(), relu6)
      Conv(1×1, 192=>32, BatchNorm())
    +
    identity

64Channels:
[6, 64, 4, 2]-1
28x32=>14x64
  Sequential
    Conv(1×1, 32=>192, BatchNorm(), relu6)
    Conv(3×3, 1=>192, padding=(1, 1), stride=(2, 2), groups=192, BatchNorm(), relu6)
    Conv(1×1, 192=>64, BatchNorm())

[6, 64, 4, 2]-2
14x64=>14x64-1
  Residual
    Sequential
      Conv(1×1, 64=>384, BatchNorm(), relu6)
      Conv(3×3, 1=>384, padding=(1, 1), groups=384, BatchNorm(), relu6)
      Conv(1×1, 384=>64, BatchNorm())
    +
    identity

[6, 64, 4, 2]-3
14x64=>14x64-2
  Residual
    Sequential
      Conv(1×1, 64=>384, BatchNorm(), relu6)
      Conv(3×3, 1=>384, padding=(1, 1), groups=384, BatchNorm(), relu6)
      Conv(1×1, 384=>64, BatchNorm())
    +
    identity

[6, 64, 4, 2]-4
14x64=>14x64-3
  Residual
    Sequential
      Conv(1×1, 64=>384, BatchNorm(), relu6)
      Conv(3×3, 1=>384, padding=(1, 1), groups=384, BatchNorm(), relu6)
      Conv(1×1, 384=>64, BatchNorm())
    +
    identity

96Channels:
[6, 96, 3, 1]-1
14x64=>14x96
  Sequential
    Conv(1×1, 64=>384, BatchNorm(), relu6)
    Conv(3×3, 1=>384, padding=(1, 1), groups=384, BatchNorm(), relu6)
    Conv(1×1, 384=>96, BatchNorm())

[6, 96, 3, 1]-2
14x96=>14x96-1
  Residual
    Sequential
      Conv(1×1, 96=>576, BatchNorm(), relu6)
      Conv(3×3, 1=>576, padding=(1, 1), groups=576, BatchNorm(), relu6)
      Conv(1×1, 576=>96, BatchNorm())
    +
    identity

[6, 96, 3, 1]-3
14x96=>14x96-2
  Residual
    Sequential
      Conv(1×1, 96=>576, BatchNorm(), relu6)
      Conv(3×3, 1=>576, padding=(1, 1), groups=576, BatchNorm(), relu6)
      Conv(1×1, 576=>96, BatchNorm())
    +
    identity

160Channels:
[6, 160, 3, 2]-1
14x96=>7x160
  Sequential
    Conv(1×1, 96=>576, BatchNorm(), relu6)
    Conv(3×3, 1=>576, padding=(1, 1), stride=(2, 2), groups=576, BatchNorm(), relu6)
    Conv(1×1, 576=>160, BatchNorm())

[6, 160, 3, 2]-2
7x160=>7x160-1
  Residual
    Sequential
      Conv(1×1, 160=>960, BatchNorm(), relu6)
      Conv(3×3, 1=>960, padding=(1, 1), groups=960, BatchNorm(), relu6)
      Conv(1×1, 960=>160, BatchNorm())
    +
    identity

[6, 160, 3, 2]-3
7x160=>7x160-2
  Residual
    Sequential
      Conv(1×1, 160=>960, BatchNorm(), relu6)
      Conv(3×3, 1=>960, padding=(1, 1), groups=960, BatchNorm(), relu6)
      Conv(1×1, 960=>160, BatchNorm())
    +
    identity

320Channels:
[6, 320, 1, 1]-1
7x160=>7x320
  Sequential
    Conv(1×1, 160=>960, BatchNorm(), relu6)
    Conv(3×3, 1=>960, padding=(1, 1), groups=960, BatchNorm(), relu6)
    Conv(1×1, 960=>320, BatchNorm())

Output:
7x320=>7x1280
  Conv(1×1, 320=>1280, BatchNorm(), relu6)

  Sequential
    x->pool(x; mode=1, window=size(x)[1:2])
    x->reshape(x, :, size(x,4))
    Linear(1280=>1000, bias, dropout=0.2)

=#
