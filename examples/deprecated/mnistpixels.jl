# This is the pixel-by-pixel MNIST problem from: Le, Q. V., Jaitly,
# N., & Hinton, G. E. (2015). A Simple Way to Initialize Recurrent
# Networks of Rectified Linear Units. arXiv preprint arXiv:1504.00941.

module MNISTPixels
using Main, Knet, ArgParse
using Knet: nextidx, stack_isempty, mat2d

isdefined(:MNIST) || include("mnist.jl")

function main(args=ARGS)
    isa(args, AbstractString) && (args=split(args))
    opts = parse_commandline(args)
    info("Pixel-by-pixel MNIST problem from Le et al. 2015.")
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)

    global trn = Pixels(mat2d(MNIST.xtrn), MNIST.ytrn; batch=batchsize, epoch=epochsize, bootstrap=true)
    global tst = Pixels(mat2d(MNIST.xtst), MNIST.ytst; batch=batchsize)
    global net = compile(:mpixels; rnn=symbol(nettype), hidden=hidden, nclass=10, winit=Gaussian(0,winit), fbias=fbias)
    setp(net; lr=lrate)

    l = [0f0,0f0]; m = [0f0,0f0]; acc = 0
    for epoch=1:epochs
        train(net, trn, softloss; gclip=gclip, losscnt=fill!(l,0), maxnorm=fill!(m,0))
        println(tuple(:trn,epoch*trn.epochsize,l[1]/l[2],m...))
        gcheck > 0 && gradcheck(net, f->gradloss(f,trn,softloss;grad=true), f->gradloss(f,trn,softloss); gcheck=gcheck)
        if epoch % testfreq == 0
            acc = 1-test(net, tst, zeroone)
            println(tuple(:tst,epoch*trn.epochsize,acc))
        end
        flush(STDOUT)
    end
    return (acc, l[1]/l[2], m...)
end

@knet function mpixels(x; rnn=nothing, hidden=0, nclass=0, o...)
    h = rnn(x; o..., out=hidden)
    if predict
        return wbf(h; o..., out=nclass, f=:soft)
    end
end

# @knet function p1(x; rnn=nothing, hidden=0, o...)
#     y = rnn(x; o..., out=hidden)
# end

# @knet function p2(x; nclass=0, o...)
#     y = wbf(x; o..., out=nclass, f=soft)
# end

function train(f, data, loss; gclip=0, losscnt=nothing, maxnorm=nothing)
    reset!(f)
    for (x,ygold) in data
        if ygold == nothing
            sforw(f, x; predict=false)
        else
            ypred = sforw(f, x; predict=true)
            losscnt[1] += loss(ypred, ygold); losscnt[2] += 1
            sback(f, ygold, loss)
            while !stack_isempty(f); sback(f); end
            g = gnorm(f); g > maxnorm[2] && (maxnorm[2]=g)
            gscale = (g > gclip > 0 ? gclip/g : 1)
            update!(f; gscale=gscale)
            w = wnorm(f); w > maxnorm[1] && (maxnorm[1]=w) # TODO: take this maxnorm calculation elsewhere, have a statistics optional function? just be smart about scaling?  batch-normalization, msra etc.
            reset!(f)
        end
    end
end

function test(f, data, loss; gclip=0, losscnt=nothing, maxnorm=nothing)
    sumloss = numloss = 0
    reset!(f)
    for (x,ygold) in data
        if ygold == nothing
            forw(f, x; predict=false)
        else
            ypred = forw(f, x; predict=true)
            sumloss += loss(ypred, ygold); numloss += 1
            reset!(f)
        end
    end
    sumloss / numloss
end

function gradloss(f, data, loss; grad=false, seed=42)
    data_rng = data.rng
    data.rng = MersenneTwister()
    srand(data.rng, seed)
    reset!(f)
    myforw = grad ? sforw : forw
    loss1 = 0
    for (x,ygold) in data
        if ygold == nothing
            myforw(f, x; predict=false)
        else
            ypred = myforw(f, x; predict=true)
            loss1 = loss(ypred, ygold)
            if grad
                sback(f, ygold, loss)
                while !stack_isempty(f); sback(f); end
            end
            break
        end
    end
    data.rng = data_rng
    return loss1
end

# Data generator:

import Base: start, next, done

# input comes in as xtrn(784,60000), ytrn(10,60000)
# the batch arg determines how many images we present in parallel
# each call to the next should output batch pixels, i.e. x(1,batch), y=nothing
# the last pixel should be served as x(1,batch), y(10,batch)

type Pixels; x; y; rng; datasize; epochsize; batchsize; bootstrap; shuffle; xbatch; ybatch; images;
    function Pixels(x, y; rng=MersenneTwister(), epoch=ccount(x), batch=16, bootstrap=false, shuffle=false)
        nx = ccount(x)
        nx == ccount(y) || error("Item count mismatch")
        shuf = (shuffle ? shuffle!(rng,[1:nx;]) : nothing)
        xbatch = similar(x, (1,batch))
        ybatch = similar(y, (clength(y),batch))
        new(x, y, rng, nx, epoch, batch, bootstrap, shuf, xbatch, ybatch, nothing)
    end
end

# state is an image/pixel pair
start(d::Pixels)=(d.shuffle != nothing && shuffle!(d.rng, d.shuffle); (0,0))

# epochsize is given as image count, not pixel count
done(d::Pixels, s)=(s[1] >= d.epochsize)

function next(d::Pixels, s)
    (n,t) = s                   # image and pixel count
    t==0 && (d.images = nextidx(d,n)) # image indices for current batch
    nb = length(d.images)       # batch size, xbatch[1:nb] needs to be filled
    nt = clength(d.x)           # pixels per image
    t += 1                      # next pixel to serve
    for b=1:nb                  # batch index
        i=d.images[b]           # image index
        d.xbatch[b] = d.x[t,i]
        t == nt && (d.ybatch[:,b] = d.y[:,i])
    end
    return t < nt ?
    ((d.xbatch, nothing), (n, t)) :
    ((d.xbatch, d.ybatch), (n+nb, 0))
end

function parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--epochs"
        help = "Number of epochs to train"
        arg_type = Int
        default = 1
        "--epochsize"
        help = "number of training examples per epoch"
        arg_type = Int
        default = 100 # 10000
        "--testfreq"
        help = "Compute test accuracy every acc epochs"
        arg_type = Int
        default = 1 # 10
        "--batchsize"
        help = "minibatch size"
        arg_type = Int
        default = 20 # 16
        "--hidden"
        help = "number of hidden units"
        arg_type = Int
        default = 100
        "--lrate"
        help = "learning rate"
        arg_type = Float64
        default = 0.005          # paper says 1e-8? 
        "--gclip"
        help = "gradient clip"
        arg_type = Float64
        default = 1.0
        "--gcheck"              # TODO: gives terrible results, check.
        help = "gradient check"
        arg_type = Int
        default = 0
        "--nettype"
        help = "type of network"
        default = "irnn" # "lstm"
        "--fbias"
        help = "forget gate bias for lstm"
        arg_type = Float64
        default =  1.0
        "--winit"
        help = "stdev for weight initialization"
        arg_type = Float64
        default =  0.01
        "--rtol"
        help = "rtol parameter for gradient checks"
        arg_type = Float64
        default =  0.01
        "--atol"
        help = "atol parameter for gradient checks"
        arg_type = Float64
        default =  0.01
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1003
    end
    parse_args(args,s)
end

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end # module

# NOTES:

# batchsize=16 is too slow.  I got good progress with the following:
# Dict{AbstractString,Any}("hidden"=>100,"rtol"=>0.01,"batchsize"=>200,"lrate"=>0.001,"train"=>10000,"gclip"=>1.0,"acc"=>10,"gcheck"=>0,"fbias"=>1.0,"epochs"=>100,"atol"=>0.01,"seed"=>1003,"type"=>"irnn")

# DONE: reconsider the data interface.
# we should make the model interface not insist on Data, any iterable should do, in fact remove the Data type.
# x field should probably be named something else like data.
# if we are ever going to need zero or more than one input, then we need to decide how to represent:
# (x1,x2,...,y) or (y,x1,x2,...): probably the first
# (y,) or y for no input tuples: probably the first

# Sample output for debugging:
# julia> include("mnistpixels.jl")
# Dict{AbstractString,Any}("hidden"=>100,"rtol"=>0.01,"batchsize"=>200,"lrate"=>0.005,"train"=>10000,"gclip"=>1.0,"acc"=>10,"gcheck"=>0,"fbias"=>1.0,"epochs"=>100,"atol"=>0.01,"seed"=>1003,"type"=>"irnn","std"=>0.01)
# (:trn,10000,2.261733271443444,10.417845f0,537.297f0)
# (:trn,20000,2.3187815377575203,10.421104f0,5327.3f0)
# (:trn,30000,2.509523073669509,10.423339f0,10088.415f0)
# (:trn,40000,2.412966892292024,10.425221f0,9402.696f0)
# (:trn,50000,2.3695122707481397,10.4281435f0,13412.184f0)
# (:trn,60000,2.268378955795277,10.4292965f0,5621.7524f0)
# (:trn,70000,2.1479991531190734,10.431348f0,2633.2693f0)
# (:trn,80000,2.1210844235681883,10.433363f0,1978.7836f0)
# (:trn,90000,2.153266387169849,10.435466f0,3289.0789f0)
# (:trn,100000,2.1011612425016724,10.437336f0,3303.134f0)
# (:tst,100000,0.1735)

# why isn't this giving the same results?
# Could be softloss vs xentloss or Float32 vs Float64.

# Here is the new sample output:
# julia> include("mnistpixels.jl")
# INFO: Loading MNIST...
#   5.863069 seconds (279.54 k allocations: 498.444 MB, 1.61% gc time)
# Dict{AbstractString,Any}("hidden"=>100,"rtol"=>0.01,"batchsize"=>200,"lrate"=>0.005,"train"=>10000,"gclip"=>1.0,"acc"=>10,"std"=>0.01,"gcheck"=>0,"fbias"=>1.0,"epochs"=>100,"atol"=>0.01,"seed"=>1003,"type"=>"irnn")
# (:trn,10000,2.262306f0,10.417742f0,537.452f0)
# (:trn,20000,2.277638f0,10.4206705f0,2240.0566f0)
# (:trn,30000,2.45132f0,10.423019f0,11971.223f0)
# (:trn,40000,2.2885273f0,10.424828f0,7045.522f0)
# (:trn,50000,2.3827329f0,10.427834f0,12723.293f0)
# (:trn,60000,2.2246392f0,10.429581f0,3678.2678f0)
# (:trn,70000,2.157727f0,10.431644f0,2867.3608f0)
# (:trn,80000,2.1695452f0,10.434076f0,4772.0884f0)
# (:trn,90000,2.2076025f0,10.435675f0,13315.624f0)
# (:trn,100000,2.174602f0,10.437442f0,7863.392f0)
# (:tst,100000,0.2939)

# sample output for lstm:
#
# [dy_052@hpc3013 examples]$ julia mnistpixels.jl --type lstm
# WARNING: requiring "Options" in module "Main" did not define a corresponding module.
# INFO: Loading MNIST...
#   6.725308 seconds (363.19 k allocations: 358.143 MB, 0.62% gc time)
# Dict{AbstractString,Any}("hidden"=>100,"rtol"=>0.01,"batchsize"=>200,"lrate"=>0.005,"train"=>10000,"gclip"=>1.0,"acc"=>10,"std"=>0.01,"gcheck"=>0,"fbias"=>1.0,"epochs"=>100,"atol"=>0.01,"seed"=>1003,"type"=>"lstm")
# (:trn,10000,2.3025522f0,14.711951f0,0.1208294f0)
# (:trn,20000,2.3024626f0,14.717022f0,0.12132876f0)
# (:trn,30000,2.3024027f0,14.721309f0,0.10962964f0)
# (:trn,40000,2.3023393f0,14.725827f0,0.14956589f0)

# S2C no longer accepts Net, it expects kfun:
    # p1 = (nettype == "irnn" ? Net(irnn; out=hidden, winit=Gaussian(0,winit)) :
    #       nettype == "lstm" ? Net(lstm; out=hidden, fbias=fbias) : 
    #       error("Unknown network type "*nettype))
    # p2 = Net(wbf; out=10, winit=Gaussian(0,winit), f=soft)
