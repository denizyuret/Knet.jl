for p in ("Knet","ArgParse","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","mnist.jl"))
include(Pkg.dir("Knet","data","imagenet.jl"))

"""

julia dcgan.jl --outdir ~/dcgan-out
julia dcgan.jl -h # to see all other script options

This example implements a DCGAN (Deep Convolutional Generative Adversarial Network) on MNIST dataset. This implemented model is not identical with the original model. LeNet is used as a base to adapt DCGAN to original MNIST data.1

* Paper url: https://arxiv.org/abs/1511.06434

"""
module DCGAN
using Knet
using Images
using ArgParse
using JLD2, FileIO

function main(args)
    o = parse_options(args)
    o[:seed] > 0 && Knet.setseed(o[:seed])

    # load models, data, optimizers
    wd, wg, md, mg = load_weights(o[:atype], o[:zdim], o[:loadfile])
    xtrn,ytrn,xtst,ytst = Main.mnist()
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; shuffle=true, xtype=o[:atype])
    optd = map(wi->eval(parse(o[:optim])), wd)
    optg = map(wi->eval(parse(o[:optim])), wg)
    z = sample_noise(o[:atype],o[:zdim],prod(o[:gridsize]))

    if o[:outdir] != nothing && !isdir(o[:outdir])
        mkpath(o[:outdir])
        mkpath(joinpath(o[:outdir],"models"))
        mkpath(joinpath(o[:outdir],"generations"))
    end

    # training
    println("training started..."); flush(STDOUT)
    for epoch = 1:o[:epochs]
        dlossval = glossval = 0
        @time for (x,y) in dtrn
            noise = sample_noise(o[:atype],o[:zdim],length(y))
            dlossval += train_discriminator!(wd,wg,md,mg,2x-1,y,noise,optd,o)
            noise = sample_noise(o[:atype],o[:zdim],length(y))
            glossval += train_generator!(wg,wd,mg,md,noise,y,optg,o)
        end
        dlossval /= length(dtrn); glossval /= length(dtrn)
        println((:epoch,epoch,:dloss,dlossval,:gloss,glossval))
        flush(STDOUT)

        # save models and generations
        if o[:outdir] != nothing
            filename = @sprintf("%04d.png",epoch)
            filepath = joinpath(o[:outdir],"generations",filename)
            plot_generations(
                wg, mg; z=z, savefile=filepath,
                scale=o[:gridscale], gridsize=o[:gridsize])

            filename = @sprintf("%04d.jld2",epoch)
            filepath = joinpath(o[:outdir],"models",filename)
            save_weights(filepath,wd,wg,md,mg)
        end
    end

    return wd,wg,md,mg
end

function parse_options(args)
    s = ArgParseSettings()
    s.description =
        "Deep Convolutional Generative Adversarial Networks on MNIST."

    @add_arg_table s begin
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}");
         help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=100; help="batch size")
        ("--zdim"; arg_type=Int; default=100; help="noise dimension")
        ("--epochs"; arg_type=Int; default=20; help="# of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--gridsize"; arg_type=Int; nargs=2; default=[8,8])
        ("--gridscale"; arg_type=Float64; default=2.0)
        ("--optim"; default="Adam(;lr=0.0002, beta1=0.5)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; default=nothing; help="output dir for models/generations")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))
    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
    end
    return o
end

function load_weights(atype,zdim,loadfile=nothing)
    if loadfile == nothing
        wd, md = initwd(atype)
        wg, mg = initwg(atype,zdim)
    else
        @load loadfile wd wg md mg
        wd = convert_weights(wd, atype)
        wg = convert_weights(wg, atype)
        md = convert_moments(md, atype)
        mg = convert_moments(mg, atype)
    end
    return wd, wg, md, mg
end

function save_weights(savefile,wd,wg,md,mg)
    save(savefile,
         "wd", convert_weights(wd),
         "wg", convert_weights(wg),
         "md", convert_moments(md),
         "mg", convert_moments(mg))
end

function convert_weights(w, atype=Array{Float32})
    w0 = map(wi->convert(atype, wi), w)
    w1 = convert(Array{Any}, w0)
end


function convert_moments(moments,atype=Array{Float32})
    clone = map(mi->bnmoments(), moments)
    for k = 1:length(clone)
        if moments[k].mean != nothing
            clone[k].mean = convert(atype, moments[k].mean)
        end

        if moments[k].var != nothing
            clone[k].var = convert(atype, moments[k].var)
        end
    end
    return convert(Array{Any,1}, clone)
end


function leaky_relu(x, alpha=0.2)
    pos = max(0,x)
    neg = min(0,x) * alpha
    return pos + neg
end

function sample_noise(atype,zdim,nsamples,mu=0.5,sigma=0.5)
    noise = convert(atype, randn(zdim,nsamples))
    normalized = (noise-mu)/sigma
end

function initwd(atype, winit=0.01)
    w = Any[]
    m = Any[]

    push!(w, winit*randn(5,5,1,20))
    push!(w, bnparams(20))
    push!(m, bnmoments())

    push!(w, winit*randn(5,5,20,50))
    push!(w, bnparams(50))
    push!(m, bnmoments())

    push!(w, winit*randn(500,800))
    push!(w, bnparams(500))
    push!(m, bnmoments())

    push!(w, winit*randn(2,500))
    push!(w, zeros(2,1))
    return convert_weights(w,atype), m
end

function dnet(w,x0,m; training=true, alpha=0.2)
    x1 = dlayer1(x0, w[1:2], m[1]; training=training)
    x2 = dlayer1(x1, w[3:4], m[2]; training=training)
    x3 = reshape(x2, 800,size(x2,4))
    x4 = dlayer2(x3, w[5:6], m[3]; training=training)
    x5 = w[end-1] * x4 .+ w[end]
end

function dlayer1(x0, w, m; stride=1, padding=0, alpha=0.2, training=true)
    x = conv4(w[1], x0; stride=stride, padding=padding)
    x = batchnorm(x, m, w[2]; training=training)
    x = leaky_relu.(x,alpha)
    x = pool(x; mode=2)
end

function dlayer2(x, w, m; training=true, alpha=0.2)
    x = w[1] * x
    x = batchnorm(x, m, w[2]; training=training)
    x = leaky_relu.(x, alpha)
end

function dloss(w,m,real_images,real_labels,fake_images,fake_labels)
    yreal = dnet(w,real_images,m)
    real_loss = nll(yreal, real_labels)
    yfake = dnet(w,fake_images,m)
    fake_loss = nll(yfake, fake_labels)
    return real_loss + fake_loss
end

dlossgradient = gradloss(dloss)

function train_discriminator!(wd,wg,md,mg,real_images,ygold,noise,optd,o)
    fake_images = gnet(wg,noise,mg; training=true)
    nsamples = div(length(real_images),784)
    real_labels = ones(Int64, 1, nsamples)
    fake_labels = 2ones(Int64, 1, nsamples)
    gradients, lossval = dlossgradient(
        wd,md,real_images,real_labels,fake_images,fake_labels)
    update!(wd, gradients, optd)
    return lossval
end

function initwg(atype=Array{Float32}, zdim=100, winit=0.01)
    w = Any[]
    m = Any[]

    # 2 dense layers combined with batch normalization layers
    push!(w, winit*randn(500,zdim))
    push!(w, bnparams(500))
    push!(m, bnmoments())

    push!(w, winit*randn(800,500)) # reshape 4x4x16
    push!(w, bnparams(800))
    push!(m, bnmoments())

    # 3 deconv layers combined with batch normalization layers
    push!(w, winit*randn(2,2,50,50))
    push!(w, bnparams(50))
    push!(m, bnmoments())

    push!(w, winit*randn(5,5,20,50))
    push!(w, bnparams(20))
    push!(m, bnmoments())

    push!(w, winit*randn(2,2,20,20))
    push!(w, bnparams(20))
    push!(m, bnmoments())

    # final deconvolution layer
    push!(w, winit*randn(5,5,1,20))
    push!(w, winit*randn(1,1,1,1))
    return convert_weights(w,atype), m
end

function gnet(wg,z,m; training=true)
    x1 = glayer1(z, wg[1:2], m[1]; training=training)
    x2 = glayer1(x1, wg[3:4], m[2]; training=training)
    x3 = reshape(x2, 4,4,50,size(x2,2))
    x4 = glayer2(x3, wg[5:6], m[3]; training=training)
    x5 = glayer3(x4, wg[7:8], m[4]; training=training)
    x6 = glayer2(x5, wg[9:10], m[5]; training=training)
    x7 = tanh.(deconv4(wg[end-1], x6) .+ wg[end])
end

function glayer1(x0, w, m; training=true)
    x = w[1] * x0
    x = batchnorm(x, m, w[2]; training=training)
    x = relu.(x)
end

function glayer2(x0, w, m; training=true)
    x = deconv4(w[1], x0; stride=2)
    x = batchnorm(x, m, w[2]; training=training)
end

function glayer3(x0, w, m; training=true)
    x = deconv4(w[1], x0)
    x = batchnorm(x, m, w[2]; training=training)
    x = relu.(x)
end

function gloss(wg,wd,mg,md,noise,ygold)
    fake_images = gnet(wg,noise,mg)
    ypred = dnet(wd,fake_images,md)
    return nll(ypred, ygold)
end

glossgradient = gradloss(gloss)

function train_generator!(wg,wd,mg,md,noise,labels,optg,o)
    ygold = ones(Int64, 1, length(labels))
    gradients, lossval = glossgradient(wg,wd,mg,md,noise,ygold)
    update!(wg,gradients,optg)
    return lossval
end

function plot_generations(
    wg, mg; z=nothing, gridsize=(8,8), scale=1.0, savefile=nothing)
    if z == nothing
        nimg = prod(gridsize)
        zdim = size(wg[1],2)
        atype = wg[1] isa KnetArray ? KnetArray{Float32} : Array{Float32}
        z = sample_noise(atype,zdim,nimg)
    end
    output = Array(0.5*(1+gnet(wg,z,mg; training=false)))
    images = map(i->output[:,:,:,i], 1:size(output,4))
    grid = Main.make_image_grid(images; gridsize=gridsize, scale=scale)
    if savefile == nothing
        display(colorview(Gray, grid))
    else
        save(savefile, grid)
    end
end

splitdir(PROGRAM_FILE)[end] == "dcgan.jl" && main(ARGS)

end # module
