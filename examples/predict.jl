using HDF5
using ArgParse
using Knet

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "x"
        help = "HDF5 file for input"
        required = true
        "net"
        help = "Comma separated list of HDF5 layer files"
        required = true
        "out"
        help = "File prefix for trained output layer files"
        required = true
        "--nogpu"
        help = "Do not use gpu"
        action = :store_true
        "--batch"
        help = "Minibatch size"
        arg_type = Int
        default = 128
    end
    args = parse_args(s)
    Knet.gpu(!args["nogpu"])
    args["nogpu"] && blas_set_num_threads(20)
    x = h5read(args["x"], "/data")
    net = map(l->Op(l), split(args["net"],','))
    gc()
    @time y = predict(net, x, batch=args["batch"])
    @time y = predict(net, x, batch=args["batch"])
    h5write(args["out"], "data", y)
end

main()
