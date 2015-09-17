# This is the MNIST Classification from a Sequence of Pixels problem
# from: Le, Q. V., Jaitly, N., & Hinton, G. E. (2015). A Simple Way to
# Initialize Recurrent Networks of Rectified Linear Units. arXiv
# preprint arXiv:1504.00941.

using CUDArt
using KUnet
using ArgParse

function parse_commandline(a=ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--epochs"
        help = "Number of epochs to train"
        arg_type = Int
        default = 1000
        "--test"
        help = "Number of test examples per epoch"
        arg_type = Int
        default = 2000
        "--train"
        help = "number of training examples per epoch"
        arg_type = Int
        default = 2000
        "--batchsize"
        help = "minibatch size"
        arg_type = Int
        default = 16
        "--hidden"
        help = "number of hidden units"
        arg_type = Int
        default = 100
        "--lr"
        help = "learning rate"
        arg_type = Float64
        default = 1e-8
        "--gclip"
        help = "gradient clip"
        arg_type = Float64
        default = 1.0
        "--gcheck"
        help = "gradient check"
        arg_type = Int
        default = 0
        "--type"
        help = "type of network"
        default = "irnn" # "lstm"
        "--fb"
        help = "forget gate bias"
        arg_type = Float64
        default =  1.0
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1003
    end
    parse_args(a,s)
end

args = parse_commandline()
println(args)
args["seed"] > 0 && setseed(args["seed"])

using KUnet: Pixels, MNIST, xentlosslayer
KUnet.loadmnist()
trn = Pixels(MNIST.xtrn, MNIST.ytrn; batch=args["batchsize"], epoch=args["train"], bootstrap=true)
tst = Pixels(MNIST.xtst, MNIST.ytst; batch=args["batchsize"], epoch=args["test"])

nx = 1
ny = 10
nh = args["hidden"]
net1 = (args["type"] == "irnn" ? irnn(nh) :
        args["type"] == "lstm" ? lstm(nh) : 
        error("Unknown network type "*args["type"]))
args["type"] == "lstm" && setparam!(net1.op[9]; init=fill!, initp=args["fb"])

net2 = xentlosslayer(ny)

net = S2C(net1, net2)
setparam!(net; lr=args["lr"])

@time for epoch=1:args["epochs"]
    (l,maxw,maxg) = train(net, trn; gclip=args["gclip"], gcheck=args["gcheck"])
    acc = accuracy(net, tst)
    println(tuple(epoch*trn.epochsize,acc,l,maxw,maxg))
    flush(STDOUT)
end

