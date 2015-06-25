module KUnet
using Compat

include("gpu.jl");	# gpu detection
@useifgpu CUDArt
@useifgpu CUBLAS
@useifgpu CUDNN  
KUnetArray=(GPU ? Union(AbstractArray,AbstractCudaArray) : AbstractArray)

include("sparse.jl");
GPU && include("cusparse.jl");
GPU && include("cumatrix.jl");

include("util.jl");	export accuracy, cpucopy, gpucopy, @date
include("param.jl");	export Param, update, setparam!
include("net.jl");	export Layer, LossLayer, Net, train, predict, forw, back, loss, loadnet, savenet

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
include("percloss.jl"); export PercLoss # deprecated

include("kernel.jl");   export Kernel, kernel # deprecated
include("poly.jl");     export Poly           # deprecated
include("rbfk.jl");     export Rbfk           # deprecated
include("perceptron.jl"); export Perceptron
include("kperceptron.jl"); export KPerceptron
#########################

end # module
