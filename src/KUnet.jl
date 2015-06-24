module KUnet
using Compat

# See if we have gpu support.  This determines whether gpu code is
# loaded, not whether it is used.  The user can control gpu use by
# changing the array type using atype.
gpuok = true
lpath = [Pkg.dir("KUnet/src")]
for l in ("libkunet", "libcuda", "libcudart", "libcublas", "libcudnn")
    isempty(find_library([l], lpath)) && (warn("Cannot find $l");gpuok=false)
end
for p in ("CUDArt", "CUBLAS", "CUDNN")
    isdir(Pkg.dir(p)) || (warn("Cannot find $p");gpuok=false)
end
const GPU = gpuok
GPU || warn("GPU libraries missing, using CPU.")
USEGPU = GPU
gpu()=USEGPU
gpu(b::Bool)=(b && !gpuok && error("No GPU"); global USEGPU=b)

# Conditional module import
macro useifgpu(pkg) if GPU Expr(:using,pkg) end end
@useifgpu CUDArt
@useifgpu CUBLAS
@useifgpu CUDNN  
# @useifgpu CUSPARSE
KUnetArray=(GPU ? Union(AbstractArray,AbstractCudaArray) : AbstractArray)

#########################
include("util.jl");	export accuracy, cpucopy, gpucopy, @date # and extends basic functions for cuda
include("cusparse.jl");

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
