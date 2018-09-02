using KUnet, HDF5, JLD, ArgParse, Compat

function initnet()
    args = parse_commandline()
    net = Op[]
    !isempty(args["dropout"]) && push!(net, Drop(args["dropout"][1]))
    for h in args["hidden"]
        append!(net, [Mmul(h), Bias(), Relu()])
        !isempty(args["dropout"]) && push!(net, Drop(args["dropout"][end]))
    end
    append!(net, [Mmul(args["nclass"]), Bias(), Logp(), LogpLoss()])
    for k in [fieldnames(Param)]
        haskey(args, string(k)) || continue
        v = args[string(k)]
        if isempty(v)
            continue
        elseif length(v)==1
            @eval setparam!($net; $k=$v[1])
        else 
            @assert length(v)==length(net) "$k should have 1 or $(length(net)) elements"
            for i=1:length(v)
                @eval setparam!($net[$i]; $k=$v[$i])
            end
        end
    end
    savenet(args["netfile"], net)
    display(net)
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "netfile"
        help = "JLD file to save the net"
        required = true
        "--nclass", "-n"
        help = "Number of output classes"
        arg_type = Int
        required = true
        "--hidden"
        help = "One or more hidden layer sizes"
        nargs = '+'
        arg_type = Int
        "--dropout"
        help = "Dropout probability"
        arg_type = Float32
        nargs = '+'
        # default = [0.1f0, 0.5f0]
        "--adagrad"
        help = "If nonzero apply adagrad using arg as epsilon"
        arg_type = Float32
        nargs = '+'
        default = [1f-8]
        "--l1reg"
        help = "L1 regularization parameter"
        arg_type = Float32
        nargs = '+'
        "--l2reg"
        help = "L2 regularization parameter"
        arg_type = Float32
        nargs = '+'
        "--lr"
        help = "Learning rate"
        arg_type = Float32
        nargs = '+'
        default = [0.01f0]
        "--maxnorm"
        help = "If nonzero upper limit on weight matrix row norms"
        arg_type = Float32
        nargs = '+'
        "--momentum"
        help = "Momentum"
        arg_type = Float32
        nargs = '+'
        "--nesterov"
        help = "Apply nesterov's accelerated gradient"
        arg_type = Float32
        nargs = '+'
    end
    args = parse_args(s)
    for (arg,val) in args
        print("$arg:$val ")
    end
    println("")
    return args
end

initnet()

