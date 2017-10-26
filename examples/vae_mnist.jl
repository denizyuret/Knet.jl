"""
Train a Variational Autoencoder on the MNIST dataset.

**Usage Example**

```julia
julia> include("vae_mnist.jl"); using Knet

julia> xytt = VAE.loadmnist(60000,10000,fashion=true,preprocess=false);
INFO: Loading FashionMNIST...

julia> VAE.main(xytt...,batchsize=100,epochs=1000,optimizer=Adam(lr=1e-3),nz=40,verb=3,infotime=1)
INFO: using Array{Float32,N} where N
(:epoch, 0, :trn, 0.6949704f0, :tst, 0.6949695f0)
(:epoch, 1, :trn, 0.33505425f0, :tst, 0.33703864f0)
elapsed time: 29.250437419 seconds
(:epoch, 2, :trn, 0.32175955f0, :tst, 0.32388464f0)
elapsed time: 30.955659897 seconds
```
"""
module VAE
using MLDatasets
using Knet
using PyPlot
import AutoGrad: getval

const F = Float32

function encode(ϕ, x)
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
    N, M = size(x)
    s = @. x * log(x̂ + F(1e-10)) + (1-x) * log(1 - x̂ + F(1e-10))
    return -sum(s) /(N*M)
end

function loss(w, x, nθ; samples=1)
    θ, ϕ = w[1:nθ], w[nθ+1:end]
    μ, logσ² = encode(ϕ, x)
    nz, M = size(μ)
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)

    KL =  - sum(@. 1 + logσ² - μ*μ - σ²) / 2
    # Normalise by same number of elements as in reconstruction
    KL /= M *28*28

    BCE = F(0)
    atype = getval(θ[1]) isa KnetArray ? KnetArray : Array
    for s=1:samples
        ϵ = convert(atype, randn(F, nz, M))
        z = @. μ + ϵ * σ
        x̂ = decode(θ, z)
        BCE += binary_cross_entropy(x, x̂)
    end
    BCE /= samples

    return BCE + KL
end

L2Reg(x) = sum(x .* x) / 2

function aveloss(θ, ϕ, data; samples=1)
    ls = F(0)
    nθ = length(θ)
    for (x, y) in data
        ls += loss([θ; ϕ], x, nθ; samples=samples)
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

function weights(nz; atype=Array{F})
    θ = [] # z->x
    nh = 400
    push!(θ, xavier(nh, nz))
    push!(θ, zeros(nh))
    push!(θ, xavier(28*28, nh)) #x
    push!(θ, zeros(28*28))
    θ = map(a->convert(atype,a), θ)

    ϕ = [] # x->z
    push!(ϕ, xavier(nh, 28*28))
    push!(ϕ, zeros(nh))
    push!(ϕ, xavier(nz, nh)) #μ
    push!(ϕ, zeros(nz))
    push!(ϕ, xavier(nz, nh)) #σ
    push!(ϕ, zeros(nz))
    ϕ = map(a->convert(atype,a), ϕ)
    return θ, ϕ
end


function loadmnist(M=60_000, Mtst=10_000; fashion=false, preprocess=false)
    if fashion
        info("Loading FashionMNIST...")
        xtrn, ytrn = FashionMNIST.traindata(1:M)
        xtst, ytst = FashionMNIST.testdata(1:Mtst)
    else
        info("Loading MNIST...")
        xtrn, ytrn = MNISTverb.traindata(1:M)
        xtst, ytst = MNIST.testdata(1:Mtst)
    end
    if preprocess
        xtrn .= (xtrn .- mean(xtrn, 3)) ./ (std(xtrn,3) .+ 1e-5)
        xtst .= (xtst .- mean(xtst, 3)) ./ (std(xtst,3) .+ 1e-5)
    end
    return xtrn, ytrn, xtst, ytst
end

function minibatch(x, y, batchsize; atype=Array{F})
    xbatch(a)=convert(atype, reshape(a, 28*28, div(length(a),28*28)))
    ybatch(a)= (a[a.==0].=10; convert(atype,
                sparse(convert(Vector{Int},a), 1:length(a), one(eltype(a)), 10,length(a))))
    data = []
    i = 0
    while i+batchsize < size(x)[end]
        xy = xbatch(x[:,:,i+1:i+batchsize]), ybatch(y[i+1:i+batchsize])
        push!(data, xy)
        i += batchsize
    end
    return data
end

function main(; M=60000, Mtst=10000, kws...)
    xytt = loadmnist(M, Mtst)
    main(xytt...; kws...)
end

function plot_reconstruction(θ, ϕ, data)
    nimg = 10
    x, _ = rand(data)
    atype = θ[1] isa KnetArray ? KnetArray : Array
    x = convert(atype, Array(x)[:,rand(1:size(x,2),nimg)])

    μ, logσ² = encode(ϕ, x)
    nz, M = size(μ)
    σ = @. exp(logσ² / 2)
    ϵ = convert(atype, randn(F, nz, M))
    z = @. μ + ϵ * σ
    x̂ = decode(θ, z)

    x̂ = Array(reshape(x̂, 28, 28, length(x̂) ÷ 28^2))
    x = Array(reshape(x, 28, 28, length(x̂) ÷ 28^2))

    fig = figure("reconstruction",figsize=(10,3))
    clf()
    for i=1:nimg
        subplot(2, nimg, i)
        imshow(x[:,:,i]', cmap="gray") #notice the transpose
        ax = gca()
        ax[:xaxis][:set_visible](false)
        ax[:yaxis][:set_visible](false)

        subplot(2, nimg, nimg+i)
        imshow(x̂[:,:,i]', cmap="gray") #notice the transpose
        ax = gca()
        ax[:xaxis][:set_visible](false)
        ax[:yaxis][:set_visible](false)
    end
    # tight_layout()
end

function plot_dream(θ)
    nimg = 20
    nh, nz = size(θ[1])
    atype = θ[1] isa KnetArray ? KnetArray : Array

    z = convert(atype, randn(F, nz, nimg))
    x̂ = decode(θ, z)

    x̂ = Array(reshape(x̂, 28, 28, length(x̂) ÷ 28^2))

    fig = figure("dream",figsize=(6,5))
    clf()
    for i=1:nimg
        subplot(4, nimg÷4, i)
        imshow(x̂[:,:,i]', cmap="gray") #notice the transpose
        ax = gca()
        ax[:xaxis][:set_visible](false)
        ax[:yaxis][:set_visible](false)
    end
end

function main(xtrn, ytrn, xtst, ytst;
        seed = -1,
        batchsize = 100,
        optimizer = Adam(lr=1e-3),
        epochs = 100,
        infotime = 1,     # report every `infotime` epochs
        verb = 2,        # plot if verb > 2
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
        samples = 1,     # number of samples in gradient estimation
        nz = 20          # encoding dimension
    )

    info("using ", atype)
    seed > 0 && srand(seed)
    dtrn = minibatch(xtrn, ytrn, batchsize; atype=atype)
    dtst = minibatch(xtst, ytst, batchsize; atype=atype)

    θ, ϕ = weights(nz, atype=atype)
    nθ = length(θ)
    opt = [deepcopy(optimizer) for _=1:length([θ; ϕ])]


    report(epoch) = begin
            println((:epoch, epoch,
                     :trn, aveloss(θ, ϕ, dtrn, samples=samples),
                     :tst, aveloss(θ, ϕ, dtst, samples=samples)))
            if verb > 2
                plot_reconstruction(θ, ϕ, dtrn)
                plot_dream(θ)
            end
        end

    report(0); tic()
    @time for epoch=1:epochs
        train!(θ, ϕ, dtrn, opt)
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
    end; toq()

    return θ, ϕ
end

end # module
