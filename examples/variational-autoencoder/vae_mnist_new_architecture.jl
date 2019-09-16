# Library includes
using Knet
using PyPlot
using AutoGrad


# General Type definitions
const F = Float32 # Data type for gpu usage
const Atype = gpu() >= 0 ? KnetArray{F} : Array{F}
const Itype = Union{KnetArray{F,4},AutoGrad.Result{KnetArray{F,4}}}
abstract type Layer end;


# Parameter definitions
nz = 20 # Encoding dimension
nh = 400 # Size of hidden layer


"""
The Convolution layer
"""
struct Conv <: Layer; w; b; f::Function; pad::Int; str::Int; end
(c::Conv)(x::Itype) = c.f.(conv4(c.w, x, padding = c.pad, stride = c.str) .+ c.b)
Conv(w1, w2, cx, cy;f = relu,pad = 1,str = 1) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f, pad, str)


"""
The DeConvolution Layer = Reverse of Conv
"""
struct DeConv <: Layer; w; b; f::Function; pad::Int; str::Int; end
(c::DeConv)(x) = c.f.(deconv4(c.w, x, padding = c.pad, stride = c.str) .+ c.b)
DeConv(w1, w2, cx, cy;f = relu,pad = 1,str = 1) = DeConv(param(w1, w2, cx, cy), param0(1, 1, cx, 1), f, pad, str)


"""
The Dense layer
"""
struct Dense <: Layer; w; b; f::Function; end
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)
Dense(i::Int, o::Int; f = relu) = Dense(param(o, i), param0(o), f)


"""
Chain of layers
"""
struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x=l(x); end; x)
(c::Chain)(x, m) = (for (index, l) in enumerate(c.layers); x = l(x, m[index]); end; x)


"""
Chain of Networks -> Autoencoder
"""
struct Autoencoder; ϕ::Chain; θ::Chain; end
function (ae::Autoencoder)(x; samples=1, β=1, F=Float32)
    z_out = ae.ϕ(x)
    μ, logσ² = z_out[1:nz, :], z_out[nz + 1:end, :]
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)

    KL  =  -sum(@. 1 + logσ² - μ * μ - σ²) / 2
    KL /= length(x)

    BCE = F(0)

    for s = 1:samples
        ϵ = convert(Atype, randn(F, size(μ)))
        z = @. μ + ϵ * σ
        x̂ = ae.θ(z)
        BCE += binary_cross_entropy(x, x̂)
    end

    BCE /= samples

    return BCE + β * KL
end

(ae::Autoencoder)(x, y) = ae(x)


function binary_cross_entropy(x, x̂)
    x = reshape(x, size(x̂))
    s = @. x * log(x̂ + F(1e-10)) + (1 - x) * log(1 - x̂ + F(1e-10))
    return -sum(s) / length(x)
end


# Definition of the Encoder
ϕ = Chain((
    Conv(3, 3, 1, 16, pad=1),
    Conv(4, 4, 16, 32, pad=1, str=2),
    Conv(3, 3, 32, 32, pad=1),
    Conv(4, 4, 32, 64, pad=1, str=2),

    x->mat(x),

    Dense(64 * 7^2, nh),
    Dense(nh, 2 * nz),
))


# Definition of the Decoder
θ = Chain((
    Dense(nz, nh),
    Dense(nh, 64 * 7^2),

    x->reshape(x, (7, 7, 64, :)),

    DeConv(4, 4, 32, 64, pad=1, str=2),
    DeConv(3, 3, 32, 32, pad=1),
    DeConv(4, 4, 16, 32, pad=1, str=2),
    DeConv(3, 3, 1, 16, f=sigm, pad=1),
))

# Initialize the autoencoder with Encoder and Decoder
ae = Autoencoder(ϕ, θ)

# Load dataset
include(Knet.dir("data", "mnist.jl"))
dtrn, dtst = mnistdata()


"""
Visualize the progress during training
"""
function cb_plot(ae, img, epoch)
    img_o = convert(Array{Float64}, img)
    img_r = convert(Array{Float64}, ae.θ(ae.ϕ(img)[1:nz, :]))

    figure("Epoch $epoch")
    clf()
    subplot(1, 2, 1)
    title("Original")
    imshow(img_o[:, :, 1, 1])
    subplot(1, 2, 2)
    title("Reproduced")
    imshow(img_r[:, :, 1, 1])
end


"""
Main function for training
Questions to: nikolas.wilhelm@tum.de
"""
function train(ae, dtrn, iters)
    img = convert(Atype, reshape(dtrn.x[:,1], (28, 28, 1, 1)))
    for epoch = 1:iters
        @time adam!(ae, dtrn)

        if (epoch % 20) == 0
            @show ae(first(dtrn)...)
            cb_plot(ae, img, epoch)
        end
    end
end

# Precompile
@info "Precompile"
ae(first(dtrn)...)
@time adam!(ae, dtrn)

# Train
@info "Start training!"
@time train(ae, dtrn, 50)
