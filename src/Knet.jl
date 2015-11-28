module Knet
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
include("util/sparse.jl");	export KUsparse, Sparse
include("util/param.jl");	export KUparam, initzero, initgaussian, initxavier
include("util/linalg.jl");
include("util/colops.jl");	export cslice!, ccopy!, cadd!, ccat!, uniq!

include("net.jl");	export Layer, LossLayer, Net, train, predict, accuracy, forw, back, loss, loadnet, savenet
include("update.jl");	export update, setparam!

include("mmul.jl");     export Mmul
include("bias.jl");	export Bias
include("conv.jl");	export Conv
include("pool.jl");	export Pool
include("drop.jl");	export Drop

include("actf.jl");	export Logp, Relu, Sigm, Soft, Tanh
include("loss.jl");	export QuadLoss, SoftLoss, LogpLoss, XentLoss, PercLoss, ScalLoss

include("kperceptron.jl"); export KPerceptron

# include("perceptron.jl"); export Perceptron # deprecated
# include("percloss.jl"); export PercLoss # deprecated
# include("kernel.jl");   export Kernel, kernel # deprecated
# include("poly.jl");     export Poly           # deprecated
# include("rbfk.jl");     export Rbfk           # deprecated
#########################

end # module
