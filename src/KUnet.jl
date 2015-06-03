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

# Conditional module import
macro useifgpu(pkg) if GPU Expr(:using,pkg) end end
@useifgpu CUDArt
@useifgpu CUBLAS
@useifgpu CUDNN  

# Atype and Ftype are the default array and element types
Ftype = Float32
Atype = (GPU ? CudaArray : Array)
ftype()=Ftype
atype()=Atype
ftype(t)=(global Ftype=t)
atype(t)=(global Atype=t)


#########################
import Base: copy, copy!, rand!, fill!, convert, reshape
include("util.jl");	# extends functions given above
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
include("percloss.jl"); export PercLoss
#########################

end # module
