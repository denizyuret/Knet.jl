using ArgParse
using HDF5
using KUnet
using CUDArt

function main()
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
        "--nogpu"
        help = "Do not use gpu"
        action = :store_true
        "--batch"
        help = "Minibatch size"
        arg_type = Int
        default = 128
    end
    args = parse_args(s)
    batch = args["batch"]
    KUnet.gpu(!args["nogpu"])
    args["nogpu"] && blas_set_num_threads(20)
    x = h5read(args["x"], "/data")
    xx = x[:,1:batch]
    y = h5read(args["y"], "/data")
    yy = y[:,1:batch]
    if !args["nogpu"] 
        xx = CudaArray(xx)
        yy = CudaArray(yy)
    end
    net = map(l->Op(l), split(args["net"],','))
    gc()
    @time KUnet.backprop(net, xx, yy)
    h5write("$(args["out"])1.h5", net[1])
    h5write("$(args["out"])2.h5", net[2])
    @time KUnet.backprop(net, xx, yy)
end

main()
