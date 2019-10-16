# Package includes
@info "Loading Packages..."
using Pkg
for p in ("Knet","PyPlot", "AutoGrad")
    haskey(Pkg.installed(),p) || Pkg.add(p)
end
using Knet, PyPlot, AutoGrad


# General Type definitions
const F = Float32 # Data type for gpu usage
const GenType = gpu() >= 0 ? KnetArray{F} : Array{F} # General type
const ConvType = gpu() >= 0 ? KnetArray{F,4} : Array{F,4} # Specific conv type
const UnionType = Union{ConvType,AutoGrad.Result{ConvType}} # Union for backprop
abstract type Layer end; # all layer types


# Parameter definitions
nz = 10 # Bottelneck
nh = 400 # Size of hidden layer
nc = 16 # Channel number in network
epochs = 20 # Number of trainig epochs
batch_size = 100 # Size of minibatch
kl_β = 1 # Beta part for kl-divergence loss


"""
The Normal Convolution layer
"""
struct Conv <: Layer; w; b; f::Function; pad::Int; str::Int; end
(c::Conv)(x::UnionType) = c.f.(conv4(c.w, x, padding = c.pad, stride = c.str) .+ c.b)
Conv(w1, w2, cx, cy;f = relu, pad=1,str=1) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f, pad, str)


"""
The Normal DeConvolution Layer = Reverse of Conv
"""
struct DeConv <: Layer; w; b; f::Function; pad::Int; str::Int; end
(c::DeConv)(x) = c.f.(deconv4(c.w, x, padding = c.pad, stride = c.str) .+ c.b)
DeConv(w1, w2, cx, cy;f = relu, pad=1,str=1) = DeConv(param(w1, w2, cx, cy), param0(1, 1, cx, 1), f, pad, str)


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
Chain of Networks - Autoencoder
"""
struct Autoencoder; ϕ::Chain; θ::Chain; end
function (ae::Autoencoder)(x; samples=1, β=kl_β, F=Float32)
    z_out = ae.ϕ(x)
    μ, logσ² = z_out[1:nz, :], z_out[nz + 1:end, :]
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)

    KL  =  -sum(@. 1 + logσ² - μ * μ - σ²) / 2
    KL /= length(x)

    BCE = F(0)

    for s = 1:samples
        ϵ = convert(GenType, randn(F, size(μ)))
        z = @. μ + ϵ * σ
        x̂ = ae.θ(z)
        BCE += binary_cross_entropy(x, x̂)
    end

    BCE /= samples

    return BCE + β * KL
end


# Autoencoder only pays attention to the first input
(ae::Autoencoder)(x, y) = ae(x)


function binary_cross_entropy(x, x̂)
    x = reshape(x, size(x̂))
    s = @. x * log(x̂ + F(1e-10)) + (1 - x) * log(1 - x̂ + F(1e-10))
    return -sum(s) / length(x)
end


# Definition of the Encoder
ϕ = Chain((
    Conv(3, 3, 1, nc, pad=1),
    Conv(4, 4, 1*nc, 2*nc, pad=1, str=2),
    Conv(3, 3, 2*nc, 2*nc, pad=1),
    Conv(4, 4, 2*nc, 4*nc, pad=1, str=2),

    x->mat(x),

    Dense(4*nc * 7^2, nh),
    Dense(nh, 2 * nz),
))


# Definition of the Decoder
θ = Chain((
    Dense(nz, nh),
    Dense(nh, 4*nc * 7^2),

    x->reshape(x, (7, 7, 4*nc, :)),

    DeConv(4, 4, 2*nc, 4*nc, pad=1, str=2),
    DeConv(3, 3, 2*nc, 2*nc, pad=1),
    DeConv(4, 4, 1*nc, 2*nc, pad=1, str=2),
    DeConv(3, 3, 1, nc, f=sigm, pad=1),
))

# Initialize the autoencoder with Encoder and Decoder
ae = Autoencoder(ϕ, θ)


# Load dataset specific functionality
include(Knet.dir("data", "mnist.jl"))
include(Knet.dir("data", "imagenet.jl"))
dtrn, dtst = mnistdata(batchsize=100)


"""
Visualize the progress during training
"""
function cb_plot(ae, imgs, epoch, dtrn; ns_img=5)
    loss = round(ae(first(dtrn)...); digits=3) # loss on 1. batch

    img_o = convert(Array{Float64}, imgs)
    img_o = map(i->reshape(img_o[:,:,:,i], (28,28,1)), 1:ns_img^2)

    img_r = convert(Array{Float64}, ae.θ(ae.ϕ(imgs)[1:nz, :]))
    img_r = map(i->reshape(img_r[:,:,:,i], (28,28,1)), 1:ns_img^2)


    figure("Training batch: $epoch, Loss: $loss")
    clf()
    subplot(1, 2, 1)
    title("Original")
    imshow(make_image_grid(img_o; gridsize=(ns_img, ns_img), scale=1))
    subplot(1, 2, 2)
    title("Reproduced")
    imshow(make_image_grid(img_r; gridsize=(ns_img, ns_img), scale=1))
end


"""
Main function for training
Questions to: nikolas.wilhelm@tum.de
"""
function train(ae, dtrn, epochs, ns_img=5; visualize=true, state_display=1000)
    imgs = convert(GenType, reshape(dtrn.x[:,1:ns_img^2], (28, 28, 1, :)))

    # Training
    for (batch, _) in progress(enumerate(adam(ae, repeat(dtrn, epochs))))
        if (batch % state_display) == 0 && visualize
            cb_plot(ae, imgs, batch, dtrn, ns_img=5) # perform callback
        end
    end
end


# Precompile
@info "Precompiling..."
@time adam!(ae, dtrn)

# Train
@info "Start training for $epochs epochs!"
@time train(ae, dtrn, epochs)
