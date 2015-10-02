# This is the pixel-by-pixel MNIST problem from: Le, Q. V., Jaitly,
# N., & Hinton, G. E. (2015). A Simple Way to Initialize Recurrent
# Networks of Rectified Linear Units. arXiv preprint arXiv:1504.00941.

using Knet
using Knet: nextidx
import Base: start, next, done
using ArgParse
include("irnn.jl")
include("lstm.jl")
include("s2c.jl")
isdefined(:MNIST) || include("mnist.jl")

function mnistpixels(args=ARGS)
    isa(args, AbstractString) && (args=split(args))
    opts = parse_commandline(args)
    info("Pixel-by-pixel MNIST problem from Le et al. 2015.")
    println(opts)
    opts["seed"] > 0 && setseed(opts["seed"])

    trn = Pixels(MNIST.xtrn, MNIST.ytrn; batch=opts["batchsize"], epoch=opts["epochsize"], bootstrap=true)
    tst = Pixels(MNIST.xtst, MNIST.ytst; batch=opts["batchsize"])

    nx = 1
    ny = 10
    p1 = (opts["type"] == "irnn" ? irnn(n=opts["hidden"], std=opts["std"]) :
          opts["type"] == "lstm" ? lstm(n=opts["hidden"], fbias=opts["fbias"]) : 
          error("Unknown network type "*opts["type"]))
    p2 = softlayer(n=10, std=opts["std"])
    net = S2C(Net(p1), Net(p2))
    setopt!(net; lr=opts["lrate"])
    l = maxw = maxg = acc = 0
    @time for epoch=1:opts["epochs"]
        (l,maxw,maxg) = train(net, trn; gclip=opts["gclip"], gcheck=opts["gcheck"], rtol=opts["rtol"], atol=opts["atol"])
        println(tuple(:trn,epoch*trn.epochsize,l,maxw,maxg))
        if epoch % opts["acc"] == 0
            acc = accuracy(net, tst)
            println(tuple(:tst,epoch*trn.epochsize,acc))
        end
        flush(STDOUT)
    end
    return (acc, l, maxw, maxg)
end

softlayer(;n=1,std=0.01) = quote
    x = input()
    w = par($n,0; init=Gaussian(0,$std))
    y = dot(w,x)
    b = par(0; init=Constant(0))
    z = add(b,y)
    l = softmax(z)
end

type Pixels; x; rng; datasize; epochsize; batchsize; bootstrap; shuffle; batch;
    function Pixels(x...; rng=MersenneTwister(), epoch=ccount(x[1]), batch=16, bootstrap=false, shuffle=false)
        nx = ccount(x[1])
        all(xi->ccount(xi)==nx, x) || error("Item count mismatch")
        idx = (shuffle ? shuffle!(rng,[1:nx;]) : nothing)
        xbatch = [ similar(x[1], (1,batch)) for i=(1:clength(x[1])) ]
        ybatch = similar(x[2], (clength(x[2]),batch))
        new(x, rng, nx, epoch, batch, bootstrap, idx, (xbatch,ybatch))
    end
end

start(d::Pixels)=(d.shuffle != nothing && shuffle!(d.rng, d.shuffle); 0)

done(d::Pixels, n)=(n >= d.epochsize)

function next(d::Pixels, n)
    idx = nextidx(d,n)
    nb = length(idx)
    nt = clength(d.x[1])
    for b=1:nb
        i=idx[b]
        t0 = (i-1)*nt
        @inbounds for t=1:nt
            d.batch[1][t][b] = d.x[1][t0 + t]
        end
        d.batch[2][:,b] = d.x[2][:,i]
    end
    (d.batch, n+nb)
end

function parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--epochsize"
        help = "number of training examples per epoch"
        arg_type = Int
        default = 1000 # 10000
        "--acc"
        help = "Compute test accuracy every acc epochs"
        arg_type = Int
        default = 1 # 10
        "--batchsize"
        help = "minibatch size"
        arg_type = Int
        default = 200 # 16
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
        "--type"
        help = "type of network"
        default = "irnn" # "lstm"
        "--fbias"
        help = "forget gate bias for lstm"
        arg_type = Float64
        default =  1.0
        "--std"
        help = "stdev for weight initialization (for irnn)"
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
        "--epochs"
        help = "Number of epochs to train"
        arg_type = Int
        default = 1
    end
    parse_args(args,s)
end

!isinteractive() && !isdefined(:load_only) && mnistpixels(ARGS)

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

# TODO: why isn't this giving the same results?
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
