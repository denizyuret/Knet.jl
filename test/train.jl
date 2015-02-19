# Options apply to all layers, weights and biases.
# (Except for regularization which applies to w, not b)
# TODO: ability to specify different learning parameters for different layers.

using HDF5
using KUnet
using ArgParse

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
    net = map(x->KUnet.Layer(x), split(args["net"],','))
    x = h5read(args["x"], "/data"); 
    y = h5read(args["y"], "/data"); 
    o = KUnet.TrainOpts()
    for (a,v) in args
        sa = symbol(a)
        if isdefined(o,sa) o.(sa) = v end
    end
    @time KUnet.train(net, x, y, o)
    out = args["out"]
    for l=1:length(net)
        h5write("$out$l.h5", net[l]);
    end
    # @time KUnet.train(net, x, y, o)
end

main()
