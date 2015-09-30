#!/bin/env julia
# Options apply to all layers, weights and biases.
# (Except for regularization which applies to w, not b)

using HDF5
using ArgParse
using Knet

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "x"
        help = "HDF5 file for input"
        required = true
        "net"
        help = "Comma separated list of HDF5 layer files"
        required = true
        "y"
        help = "HDF5 file for desired output"
        required = true
        "out"
        help = "File prefix for trained output layer files"
        required = true
        "--batch"
        help = "Minibatch size"
        arg_type = Int
        default = 128
        "--adagrad"
        help = "If nonzero apply adagrad using arg as epsilon"
        arg_type = Float32
        default = 0f0
        "--dropout"
        help = "Dropout probability"
        arg_type = Float32
        default = 0f0
        "--iters"
        help = "If nonzero limits number of batches"
        arg_type = Int
        default = 0
        "--l1reg"
        help = "L1 regularization parameter"
        arg_type = Float32
        default = 0f0
        "--l2reg"
        help = "L2 regularization parameter"
        arg_type = Float32
        default = 0f0
        "--learningRate"
        help = "Learning rate"
        arg_type = Float32
        default = 0.01f0
        "--loss"
        help = "Loss function"
        arg_type = String
        default = "softmaxloss"
        "--maxnorm"
        help = "If nonzero upper limit on weight matrix row norms"
        arg_type = Float32
        default = 0f0
        "--momentum"
        help = "Momentum"
        arg_type = Float32
        default = 0f0
        "--nesterov"
        help = "Apply nesterov's accelerated gradient"
        arg_type = Float32
        default = 0f0
        "--nogpu"
        help = "Do not use gpu"
        action = :store_true
    end
    args = parse_args(s)
    for (arg,val) in args
        print("$arg:$val ")
    end
    println("")
    return args
end

function main()
    args = parse_commandline()
    Knet.gpu(!args["nogpu"])
    args["nogpu"] && blas_set_num_threads(20)
    x = h5read(args["x"], "/data"); 
    y = h5read(args["y"], "/data"); 
    net = map(l->Op(l), split(args["net"],','))
    for (a,v) in args
        if !in(a, ["x","y","nogpu","net","batch","iters","loss","out"])
            setparam!(net, symbol(a), v)
        end
    end
    @time train(net, x, y; batch=args["batch"], iters=args["iters"], loss=eval(parse(args["loss"])))
    out = args["out"]
    for l=1:length(net)
        h5write("$out$l.h5", net[l]);
    end
end

main()
