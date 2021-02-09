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
