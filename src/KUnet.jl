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

include("util/deepcopy.jl");	export cpucopy, gpucopy
include("util/array.jl");	export BaseArray, csize, ccount, clength, atype, csub, cget, size2
include("util/dense.jl");	export KUdense
include("util/sparse.jl");	export KUsparse
include("util/linalg.jl");
include("util/colops.jl");	export cslice!, ccopy!, cadd!, ccat!, uniq!

include("util/param.jl");	export KUparam, setparam! # TODO: move this up to src
include("update.jl");	export update
include("data.jl");	export Data, ItemTensor, AddingData, TrainMNIST, TestMNIST
include("model.jl");	export Model, train, predict, test, gradcheck
include("op.jl");	export Op, forw, back, loss, params, ninputs, ysize, overwrites, back_reads_x, back_reads_y
include("net.jl");	export Net, init, nops, op  # TODO: do we still need nops and op after params?

include("mmul.jl");     export Mmul
include("bias.jl");	export Bias
include("conv.jl");	export Conv
include("pool.jl");	export Pool
include("drop.jl");	export Drop
include("add2.jl");	export Add2
include("mul2.jl");	export Mul2

include("actf.jl");	export ActfLayer, Logp, Relu, Sigm, Soft, Tanh
include("loss.jl");	export LossLayer, QuadLoss, SoftLoss, LogpLoss, XentLoss, PercLoss, ScalLoss

include("arch.jl");	export lstm, irnn, S2C
include("mlp.jl");	export MLP, accuracy, loadnet, savenet

include("kperceptron.jl"); export KPerceptron

#########################

end # module
