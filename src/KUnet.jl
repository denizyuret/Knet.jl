module KUnet
using Compat

# Print date, expression and elapsed time after execution
VERSION < v"0.4-" && eval(Expr(:using,:Dates))
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end
export @date

include("util/gpu.jl");	export gpumem, gpusync, setseed
@useifgpu CUDArt
@useifgpu CUBLAS
@useifgpu CUDNN  
GPU && include("util/cudart.jl");
GPU && include("util/curand.jl");

# TODO: minimize exports

include("util/deepcopy.jl");	export cpucopy, gpucopy
include("util/array.jl");	export BaseArray, csize, ccount, clength, atype, csub, cget, size2 # TODO: move these to colops
include("util/dense.jl");	export KUdense
include("util/sparse.jl");	export KUsparse
include("util/linalg.jl");
include("util/colops.jl");	export cslice!, ccopy!, cadd!, ccat!, uniq!

include("param.jl");		export KUparam, setparam! # TODO: move this up to src
include("update.jl");		export update!
include("data.jl");		export Data, ItemTensor
include("model.jl");		export Model, train, test, accuracy # TODO: add predict, load, save
include("op.jl");		export Op  # , forw, back, loss, params, ninputs, ysize, overwrites, back_reads_x, back_reads_y
include("net.jl");		export Net # , init, nops, op  # TODO: do we still need nops and op after params?
include("mlp.jl");		export MLP, predict # , accuracy, loadnet, savenet

include("op/mmul.jl");     	export Mmul
include("op/bias.jl");		export Bias
include("op/conv.jl");		export Conv
include("op/pool.jl");		export Pool
include("op/drop.jl");		export Drop
include("op/add2.jl");		export Add2
include("op/mul2.jl");		export Mul2
include("op/actf.jl");		export ActfLayer, Logp, Relu, Sigm, Soft, Tanh # TODO: rename -> Actf
include("op/loss.jl");		export LossLayer, QuadLoss, SoftLoss, LogpLoss, XentLoss, PercLoss, ScalLoss # TODO: rename -> Loss

include("model/irnn.jl");	export IRNN
include("model/lstm.jl");	export LSTM
include("model/s2c.jl");	export S2C
include("model/kperceptron.jl"); export KPerceptron

include("data/adding.jl");	export Adding
include("data/mnist.jl");	export MNIST
include("data/pixels.jl");	export Pixels

end # module
