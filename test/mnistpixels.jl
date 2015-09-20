# This is the MNIST Classification from a Sequence of Pixels problem
# from: Le, Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to
# Initialize Recurrent Networks of Rectified Linear Units. arXiv
# preprint arXiv:1504.00941.

using KUnet
using ArgParse

function parse_commandline(a=ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--train"
        help = "number of training examples per epoch"
        arg_type = Int
        default = 10000
        "--acc"
        help = "Compute test accuracy every acc epochs"
        arg_type = Int
        default = 10
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
        default = 100
    end
    parse_args(a,s)
end

args = parse_commandline(isdefined(:myargs) ? split(myargs) : ARGS)
println(args)
args["seed"] > 0 && setseed(args["seed"])

(dtrn,dtst)=MNIST()
trn = Pixels(dtrn.data[1], dtrn.data[2]; batch=args["batchsize"], epoch=args["train"], bootstrap=true)
tst = Pixels(dtst.data[1], dtst.data[2]; batch=args["batchsize"])

# TODO: reconsider the data interface.
# we should make the model interface not insist on Data, any iterable should do, in fact remove the Data type.
# x field should probably be named something else like data.
# if we are ever going to need zero or more than one input, then we need to decide how to represent:
# (x1,x2,...,y) or (y,x1,x2,...): probably the first
# (y,) or y for no input tuples: probably the first

nx = 1
ny = 10
nh = args["hidden"]
net1 = (args["type"] == "irnn" ? IRNN(nh; std=args["std"]) :
        args["type"] == "lstm" ? LSTM(nh; fbias=args["fbias"]) : 
        error("Unknown network type "*args["type"]))

net2 = Net(Mmul(ny), Bias(), XentLoss())

net = S2C(net1, net2)
setparam!(net; lr=args["lrate"])

@time for epoch=1:args["epochs"]
    (l,maxw,maxg) = train(net, trn; gclip=args["gclip"], gcheck=args["gcheck"], rtol=args["rtol"], atol=args["atol"])
    println(tuple(:trn,epoch*trn.epochsize,l,maxw,maxg))
    if epoch % args["acc"] == 0
        acc = accuracy(net, tst)
        println(tuple(:tst,epoch*trn.epochsize,acc))
    end
    flush(STDOUT)
end

# batchsize=16 is too slow.  I got good progress with the following:
# Dict{AbstractString,Any}("hidden"=>100,"rtol"=>0.01,"batchsize"=>200,"lrate"=>0.001,"train"=>10000,"gclip"=>1.0,"acc"=>10,"gcheck"=>0,"fbias"=>1.0,"epochs"=>100,"atol"=>0.01,"seed"=>1003,"type"=>"irnn")

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
