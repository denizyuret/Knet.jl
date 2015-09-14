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
include("util/array.jl");	export BaseArray, csize, ccount, clength, atype
include("util/dense.jl");	export KUdense
include("util/sparse.jl");	export KUsparse
include("util/param.jl");	export KUparam
include("util/linalg.jl");
include("util/colops.jl");	export cslice!, ccopy!, cadd!, ccat!, uniq!

include("net.jl");	export Layer, Net, train, predict, accuracy, forw, back, loss, loadnet, savenet, ninputs, overwrites, back_reads_x, back_reads_y, ysize, param
include("update.jl");	export update, setparam!
include("model.jl");	export Model, train, predict, test, gradcheck, setparam!

include("mmul.jl");     export Mmul
include("bias.jl");	export Bias
include("conv.jl");	export Conv
include("pool.jl");	export Pool
include("drop.jl");	export Drop
include("add2.jl");	export Add2
include("mul2.jl");	export Mul2

include("actf.jl");	export ActfLayer, Logp, Relu, Sigm, Soft, Tanh
include("loss.jl");	export LossLayer, QuadLoss, SoftLoss, LogpLoss, XentLoss, PercLoss, ScalLoss

include("rnn.jl");	export RNN, init, nops, op
include("arch.jl");	export lstm, irnn, RNN2, gradcheck
# include("train.jl");	export train, test, batch, 

include("kperceptron.jl"); export KPerceptron

# include("perceptron.jl"); export Perceptron # deprecated
# include("percloss.jl"); export PercLoss # deprecated
# include("kernel.jl");   export Kernel, kernel # deprecated
# include("poly.jl");     export Poly           # deprecated
# include("rbfk.jl");     export Rbfk           # deprecated
#########################

end # module
