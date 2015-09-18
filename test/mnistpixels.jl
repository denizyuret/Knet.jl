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
        default = 1e-3          # TODO: paper says 1e-8, should we use doubles?
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
trn = Pixels(dtrn.x[1], dtrn.x[2]; batch=args["batchsize"], epoch=args["train"], bootstrap=true)
tst = Pixels(dtst.x[1], dtst.x[2]; batch=args["batchsize"])

# TODO: reconsider the data interface.
# we should make the model interface not insist on Data, any iterable should do, in fact remove the Data type.
# x field should probably be named something else like data.
# if we are ever going to need zero or more than one input, then we need to decide how to represent:
# (x1,x2,...,y) or (y,x1,x2,...): probably the first
# (y,) or y for no input tuples: probably the first

nx = 1
ny = 10
nh = args["hidden"]
net1 = (args["type"] == "irnn" ? IRNN(nh) :
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
