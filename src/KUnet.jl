module KUnet
using Compat

# Print date, expression and elapsed time after execution
VERSION < v"0.4-" && eval(Expr(:using,:Dates))
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end

include("util/gpu.jl");	export gpumem, gpusync, setseed
@useifgpu CUDArt
@useifgpu CUBLAS
@useifgpu CUDNN  
GPU && include("util/cudart.jl");
GPU && include("util/curand.jl");

include("util/deepcopy.jl");	export cpucopy, gpucopy
include("util/array.jl");	export accuracy
include("util/dense.jl");	export KUdense, cslice!, ccopy!, ccat!
include("util/param.jl");	export KUparam
# include("util/sparse.jl");
include("util/linalg.jl");

include("net.jl");	export Layer, LossLayer, Net, train, predict, forw, back, loss, loadnet, savenet
include("update.jl");	export update, setparam!

include("bias.jl");	export Bias
include("conv.jl");	export Conv
include("drop.jl");	export Drop
include("logp.jl");	export Logp
include("mmul.jl");     export Mmul
include("pool.jl");	export Pool
include("relu.jl");	export Relu
include("sigm.jl");	export Sigm
include("soft.jl");	export Soft
include("tanh.jl");	export Tanh

include("logploss.jl");	export LogpLoss
include("quadloss.jl");	export QuadLoss
include("softloss.jl");	export SoftLoss
include("xentloss.jl");	export XentLoss

# include("percloss.jl"); export PercLoss # deprecated
# include("kernel.jl");   export Kernel, kernel # deprecated
# include("poly.jl");     export Poly           # deprecated
# include("rbfk.jl");     export Rbfk           # deprecated
# include("perceptron.jl"); export Perceptron
# include("kperceptron.jl"); export KPerceptron
#########################

end # module
