for p in ("Knet","ArgParse","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

"""
Train a Variational Autoencoder on the MNIST dataset.
"""
module VAE
using Knet
using ArgParse
using Images
include(Pkg.dir("Knet","data","mnist.jl"))
include(Pkg.dir("Knet","data","imagenet.jl"))

const F = Float32

function encode(ϕ, x)
    x = mat(x)
    x = relu.(ϕ[1]*x .+ ϕ[2])
    μ = ϕ[3]*x .+ ϕ[4]
    logσ² = ϕ[5]*x .+ ϕ[6]
    return μ, logσ²
end

function decode(θ, z)
    z = relu.(θ[1]*z .+ θ[2])
    return sigm.(θ[3]*z .+ θ[4])
end

function binary_cross_entropy(x, x̂)
    s = @. x * log(x̂ + F(1e-10)) + (1-x) * log(1 - x̂ + F(1e-10))
    return -mean(s)
end

function loss(w, x, nθ)
    θ, ϕ = w[1:nθ], w[nθ+1:end]
    μ, logσ² = encode(ϕ, x)
    nz, M = size(μ)
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)

    KL =  -sum(@. 1 + logσ² - μ*μ - σ²) / 2
    # Normalise by same number of elements as in reconstruction
    KL /= M*28*28

    z = μ .+ randn!(similar(μ)) .* σ
    x̂ = decode(θ, z)
    BCE = binary_cross_entropy(mat(x), x̂)

    return BCE + KL
end

function aveloss(θ, ϕ, data)
    ls = F(0)
    nθ = length(θ)
    for (x, y) in data
        ls += loss([θ; ϕ], x, nθ)
    end
    return ls / length(data)
end

function train!(θ, ϕ, data, opt; epochs=1)
    w = [θ; ϕ]
    for epoch=1:epochs
        for (x, y) in data
            dw = grad(loss)(w, x, length(θ))
            update!(w, dw, opt)
        end
    end
    return θ, ϕ
end

function weights(nz, nh; atype=Array{F})
    θ = [  # z->x
        xavier(nh, nz),
        zeros(nh),
        xavier(28*28, nh), #x
        zeros(28*28)
        ]
    θ = map(a->convert(atype,a), θ)

    ϕ = [ # x->z
        xavier(nh, 28*28),
        zeros(nh),
        xavier(nz, nh), #μ
        zeros(nz),
        xavier(nz, nh), #σ
        zeros(nz)
        ]
    ϕ = map(a->convert(atype,a), ϕ)

    return θ, ϕ
end

function plot_dream(θ; gridsize=(5,5), scale=1.0)
    nh, nz = size(θ[1])
    atype = θ[1] isa KnetArray ? KnetArray : Array
    m, n = gridsize
    nimg = m*n
    z = convert(atype, randn(F, nz, nimg))
    x̂ = Array(decode(θ, z))
    images = map(i->reshape(x̂[:,i], (28,28,1)), 1:nimg)
    grid = make_image_grid(images; gridsize=gridsize, scale=scale)
    display(colorview(Gray, grid))
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
        ("--lr"; arg_type=Float64; default=1e-3; help="learning rate")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{F}" : "Array{F}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--infotime"; arg_type=Int; default=2; help="report every infotime epochs")
    end
    isa(args, String) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s; as_symbols=true)

    atype = eval(parse(o[:atype]))
    info("using ", atype)
    o[:seed] > 0 && srand(o[:seed])
    atype <: KnetArray && rand!(KnetArray(ones(10))) # bug #181 of Knet

    θ, ϕ = weights(o[:nz], o[:nh], atype=atype)
    w = [θ; ϕ]
    opt = optimizers(w, Adam, lr=o[:lr])

    xtrn, ytrn, xtst, ytst = mnist()


    report(epoch) = begin
            dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=atype)
            dtst = minibatch(xtst, ytst, o[:batchsize]; xtype=atype)
            println((:epoch, epoch,
                     :trn, aveloss(θ, ϕ, dtrn),
                     :tst, aveloss(θ, ϕ, dtst)))
        end

    report(0); tic()
    @time for epoch=1:o[:epochs]
        for (x, y) in  minibatch(xtrn, ytrn, o[:batchsize], shuffle=true, xtype=atype)
            dw = grad(loss)(w, x, length(θ))
            update!(w, dw, opt)
        end
        (epoch % o[:infotime] == 0) && (report(epoch); toc(); tic())
    end; toq()

    return θ, ϕ
end

PROGRAM_FILE == "vae_mnist.jl" && main(ARGS)

end # module
