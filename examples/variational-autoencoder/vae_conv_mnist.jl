# using Pkg; for p in ("Knet","ArgParse","Images"); haskey(Pkg.installed(),p) || Pkg.add(p); end

"""
Train a Variational Autoencoder with convolutional layers
on the MNIST dataset.
"""
module VAE
using Knet, CUDA, ArgParse, Images, Random, Statistics
import AutoGrad: getval

include(Knet.dir("data","mnist.jl"))
include(Knet.dir("data","imagenet.jl"))

const F = Float32

# global variables reset in main
Atype = gpu() >= 0 ? KnetArray{F} : Array{F}
BINARIZE = false

binarize(x) = convert(Atype, rand(F, size(x))) .< x

function encode(ϕ, x)
    x = reshape(x, (28,28,1,:))

    x = conv4(ϕ[1], x, padding=1)
    x = relu.(x .+ ϕ[2])

    x = conv4(ϕ[3], x, padding=1, stride=2)
    x = relu.(x .+ ϕ[4])
    x = conv4(ϕ[5], x, padding=1)
    x = relu.(x .+ ϕ[6])

    x = conv4(ϕ[7], x, padding=1, stride=2)
    x = relu.(x .+ ϕ[8])

    x = mat(x)
    x = relu.(ϕ[9]*x .+ ϕ[10])

    μ = ϕ[end-3]*x .+ ϕ[end-2]
    logσ² = ϕ[end-1]*x .+ ϕ[end]

    return μ, logσ²
end

function decode(θ, z)
    z = relu.(θ[1]*z .+ θ[2])
    z = relu.(θ[3]*z .+ θ[4])

    filters = size(θ[5], 4)
    width = Int(sqrt(size(z,1) ÷ filters))
    z = reshape(z, (width, width, filters, :))

    z = deconv4(θ[5], z, padding=1, stride=2)
    z = relu.(z .+ θ[6])

    z = deconv4(θ[7], z, padding=1)
    z = relu.(z .+ θ[8])

    z = deconv4(θ[9], z, padding=1, stride=2)
    z = relu.(z .+ θ[10])

    z = deconv4(θ[11], z, padding=1)
    z = sigm.(z .+ θ[12])

    return z
end

function binary_cross_entropy(x, x̂)
    x = reshape(x, size(x̂))
    s = @. x * log(x̂ + F(1e-10)) + (1-x) * log(1 - x̂ + F(1e-10))
    return -sum(s) / length(x)
end

function loss(w, x; samples=1)
    θ, ϕ = w[:decoder], w[:encoder]
    μ, logσ² = encode(ϕ, x)
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)

    KL =  - sum(@. 1 + logσ² - μ*μ - σ²) / 2
    # Normalise by same number of elements as in reconstruction
    KL /= length(x)

    BCE = F(0)
    for s=1:samples
        # ϵ = randn!(similar(μ))
        ϵ = convert(Atype, randn(F, size(μ)))
        z = @. μ + ϵ * σ
        x̂ = decode(θ, z)
        BCE += binary_cross_entropy(x, x̂)
    end
    BCE /= samples

    return BCE + KL
end

lossgradient = grad(loss)

function aveloss(w, xtrn; samples=1, batchsize=100)
    θ, ϕ = w[:decoder], w[:encoder]
    ls = F(0)
    count = 0
    for x in minibatch(xtrn, batchsize; xtype=Atype)
        BINARIZE && (x = binarize(x))
        ls += loss(w, x; samples=samples)
        count += 1
    end
    N = length(xtrn) ÷ size(xtrn, ndims(xtrn))
    return (ls / count) * N
end

function weights(nz, nh)
    θ = [] # z->x

    push!(θ, xavier(nh, nz))
    push!(θ, zeros(nh))

    push!(θ, xavier(64*7^2, nh))
    push!(θ, zeros(64*7^2))

    push!(θ, xavier(4, 4, 32, 64))
    push!(θ, zeros(1,1,32,1))

    push!(θ, xavier(3, 3, 32, 32))
    push!(θ, zeros(1,1,32,1))

    push!(θ, xavier(4, 4, 16, 32))
    push!(θ, zeros(1, 1, 16, 1))

    push!(θ, xavier(3, 3, 1, 16))
    push!(θ, zeros(1,1,1,1))


    θ = map(a->convert(Atype,a), θ)

    ϕ = [] # x->z

    push!(ϕ, xavier(3, 3, 1, 16))
    push!(ϕ, zeros(1, 1, 16, 1))

    push!(ϕ, xavier(4, 4,  16, 32))
    push!(ϕ, zeros(1, 1, 32, 1))

    push!(ϕ, xavier(3, 3, 32, 32))
    push!(ϕ, zeros(1,1, 32, 1))

    push!(ϕ, xavier(4, 4, 32, 64))
    push!(ϕ, zeros(1, 1, 64, 1))

    push!(ϕ, xavier(nh, 64*7^2))
    push!(ϕ, zeros(nh))

    push!(ϕ, xavier(nz, nh)) # μ
    push!(ϕ, zeros(nz))
    push!(ϕ, xavier(nz, nh)) # logσ^2
    push!(ϕ, zeros(nz))

    ϕ = map(a->convert(Atype,a), ϕ)

    return θ, ϕ
end

function plot_dream(w; gridsize=(5,5), scale=1.0, outfile=nothing)
    θ = w[:decoder]
    nh, nz = size(θ[1])
    atype = θ[1] isa KnetArray ? KnetArray : Array
    m, n = gridsize
    nimg = m*n
    z = convert(atype, randn(F, nz, nimg))
    x̂ = Array(decode(θ, z))
    images = map(i->reshape(x̂[:,:,:,i], (28,28,1)), 1:nimg)
    grid = make_image_grid(images; gridsize=gridsize, scale=scale)

    if outfile == nothing
        display(colorview(Gray, grid))
    else
        save(outfile, colorview(Gray, grid))
    end
end

function main(args="")
    s = ArgParseSettings()
    s.description="Variational Auto Encoder on MNIST dataset."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=100; help="minibatch size")
        ("--epochs"; arg_type=Int; default=100; help="number of epochs for training")
        ("--nh"; arg_type=Int; default=400; help="hidden layer dimension")
        ("--nz"; arg_type=Int; default=40; help="encoding dimention")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{F}" : "Array{F}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--infotime"; arg_type=Int; default=1; help="report every infotime epochs")
        ("--binarize"; arg_type=Bool; default=false; help="dinamically binarize during training")
        ("--optim"; default="Adam()"; help="optimizer")
    end
    isa(args, String) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s; as_symbols=true)

    global Atype = eval(Meta.parse(o[:atype]))
    global BINARIZE = o[:binarize]
    @info("using $Atype")
    # gc(); knetgc();
    o[:seed] > 0 && Knet.seed!(o[:seed])

    xtrn, ytrn, xtst, ytst = mnist()
    wdec, wenc = weights(o[:nz], o[:nh])
    w = Dict(
        :encoder => wenc,
        :decoder => wdec)
    opt = Dict(
        :encoder => map(wi->eval(Meta.parse(o[:optim])), w[:encoder]),
        :decoder => map(wi->eval(Meta.parse(o[:optim])), w[:decoder]),
    )

    report(epoch) = begin
            println((:epoch, epoch,
                     :trn, aveloss(w, xtrn; batchsize=o[:batchsize]),
                     :tst, aveloss(w, xtst; batchsize=o[:batchsize])))
    end

    report(0)
    @time for epoch=1:o[:epochs]
        @time for x  in minibatch(xtrn, o[:batchsize]; xtype=Atype, shuffle=true)
            BINARIZE && (x = binarize(x))
            dw = lossgradient(w, x)
            update!(w, dw, opt)
        end
        (epoch % o[:infotime] == 0) && report(epoch)
    end

    return w
end

PROGRAM_FILE == "vae_conv_mnist.jl" && main(ARGS)

end # module
