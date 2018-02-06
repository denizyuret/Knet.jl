module VAE
using Knet
using Plots; gr()
import AutoGrad: getval

include(joinpath(Pkg.dir("Knet"), "data", "mnist.jl"))

const F = Float32
Atype = gpu() >= 0 ? KnetArray{F} : Array{F}

function selu(x)
    alpha = F(1.6732632)
    scale = F(1.0507009)
    p = relu(x)
    m = -relu(-x)
    return scale*(p + alpha*exp(m)-alpha)
end

act = selu 

function encode(ϕ, x)
    x = reshape(x, (28,28,1,:))

    x = conv4(ϕ[1], x, padding=1)
    x = act.(x .+ ϕ[2])
    
    x = conv4(ϕ[3], x, padding=1, stride=2)
    x = act.(x .+ ϕ[4])
    x = conv4(ϕ[5], x, padding=1)
    x = act.(x .+ ϕ[6])
    
    x = conv4(ϕ[7], x, padding=1, stride=2)
    x = act.(x .+ ϕ[8])
    
    x = mat(x)
    x = act.(ϕ[9]*x .+ ϕ[10])
    
    μ = ϕ[end-3]*x .+ ϕ[end-2]
    logσ² = ϕ[end-1]*x .+ ϕ[end]
    
    return μ, logσ²
end

function decode(θ, z)
    z = act.(θ[1]*z .+ θ[2])
    z = act.(θ[3]*z .+ θ[4])

    filters = size(θ[5], 4)
    width = Int(sqrt(size(z,1) ÷ filters))
    z = reshape(z, (width, width, filters, :))
    
    z = deconv4(θ[5], z, padding=1, stride=2)
    z = act.(z .+ θ[6])

    z = deconv4(θ[7], z, padding=1)
    z = act.(z .+ θ[8])

    z = deconv4(θ[9], z, padding=1, stride=2)
    z = act.(z .+ θ[10])
    
    z = deconv4(θ[11], z, padding=1)
    z = sigm.(z .+ θ[12])

    return z
end

function binary_cross_entropy(x, x̂)
    x = reshape(x, size(x̂))
    s = @. x * log(x̂ + F(1e-10)) + (1-x) * log(1 - x̂ + F(1e-10))
    return -sum(s) / length(x)
end

function loss(w, x, nθ; samples=1)
    θ, ϕ = w[1:nθ], w[nθ+1:end]
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

function aveloss(θ, ϕ, xtrn; samples=1, batchsize=100)
    ls = F(0)
    nθ = length(θ)
    count = 0 
    for x in minibatch(xtrn, batchsize; xtype=Atype)
        ls += loss([θ; ϕ], x, nθ; samples=samples)
        count += 1
    end
    N = length(θ[end]) 
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

function reconstruct(θ, ϕ, x)
    μ, logσ² = encode(ϕ, x)
    σ = @. exp(logσ² / 2)
    # ϵ = randn!(similar(μ)) #slow?
    ϵ = convert(Atype, randn(F, size(μ)))
    z = @. μ + ϵ * σ
    x̂ = decode(θ, z)
end

function plot_reconstruction(θ, ϕ, xtrn; outfile="")
    nimg = 10
    xtrn = reshape(xtrn, (28,28,:))
    x = xtrn[:,:,1:nimg]
    x = convert(Atype, x)

    x̂ = reconstruct(θ, ϕ, x)
    x = Array(reshape(x, (28, 28, :)))
    x̂ = Array(reshape(x̂, (28, 28, :)))

    img = vcat(hcat((x[:,:,i]' for i=1:nimg)...),
               hcat((x̂[:,:,i]' for i=1:nimg)...))
    img = flipdim(img,1)               
               
    fig = heatmap(img, 
legend=false,grid=false,border=false,ticks=false,color=cgrad(:gray,:cmocean))
    savefig(fig, outfile)
end

function plot_dream(θ; outfile="")
    nimg = 16
    nh, nz = size(θ[1])
    z = convert(Atype, randn(F, nz, nimg))
    x̂ = decode(θ, z)
    x̂ = Array(reshape(x̂, (28, 28, :)))
    
    img = vcat(hcat((x̂[:,:,i]' for i=1:4)...),
               hcat((x̂[:,:,i]' for i=5:8)...),
               hcat((x̂[:,:,i]' for i=9:12)...),
               hcat((x̂[:,:,i]' for i=13:16)...))
    img = flipdim(img,1)               

    fig = heatmap(img, 
legend=false,grid=false,border=false,ticks=false,color=cgrad(:gray,:cmocean))
    savefig(fig, outfile)
end

function main(;
        seed = -1,
        batchsize = 100,
        lr = 1e-3,
        epochs = 1000,
        infotime = 1,     # report every `infotime` epochs
        verb = 2,        # plot if verb > 2
        atype = Atype,
        samples = 1,     # number of samples in gradient estimation
        nz = 100,          # encoding dimension
        nh = 500
    )

    info("using ", atype)
    global Atype = atype
    # gc(); knetgc(); 
    seed > 0 && setseed(seed)

    xtrn, ytrn, xtst, ytst = mnist()
    θ, ϕ = weights(nz, nh)
    w = [θ; ϕ]
    nθ = length(θ)
    opt = [Adam(lr=lr) for _=1:length([θ; ϕ])]

    report(epoch) = begin
            println((:epoch, epoch,
                     :trn, aveloss(θ, ϕ, xtrn, samples=samples),
                     :tst, aveloss(θ, ϕ, xtst, samples=samples)))
            if verb > 2
                plot_reconstruction(θ, ϕ, xtrn, outfile="res/reconstr_$epoch.png")
                plot_dream(θ, outfile="res/dream_$epoch.png")
            end
        end

    report(0); tic()
    @time for epoch=1:epochs
        for x  in minibatch(xtrn, batchsize; xtype=atype, shuffle=true)
            dw = grad(loss)(w, x, length(θ))
            update!(w, dw, opt)
        end    
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
    end; toq()

    return θ, ϕ
end

end # module

