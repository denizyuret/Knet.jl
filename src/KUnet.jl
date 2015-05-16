module KUnet
using Compat
using InplaceOps
using Base.LinAlg.BLAS
using HDF5, JLD

# See if we have gpu support.  This determines whether gpu code is
# loaded, not whether it is used.  The user can control gpu use by
# changing the array type using atype.
gpuok = true
lpath = [Pkg.dir("KUnet/cuda")]
for l in ("libkunet", "libcuda", "libcudart", "libcublas", "libcudnn")
    isempty(find_library([l], lpath)) && (gpuok=false)
end
for p in ("CUDArt", "CUBLAS", "CUDNN")
    isdir(Pkg.dir(p)) || (gpuok=false)
end
const GPU = gpuok
GPU || warn("GPU libraries missing, using CPU.")

# Conditional module import
macro useifgpu(pkg) if GPU Expr(:using,pkg) end end
@useifgpu CUDArt
@useifgpu CUBLAS
@useifgpu CUDNN  

# Atype and Ftype are the default array and element types
# TODO: test this on cpu-only machine
Ftype = Float32
Atype = (GPU ? CudaArray : Array)
ftype()=Ftype
atype()=Atype
ftype(t)=(global Ftype=t)
atype(t)=(global Atype=t)

#########################
# TODO: clean util.jl, minimize cuda code
include("util.jl");	# extends copy!, mul!, badd!, bmul!, bsub!, sum!, zeros, rand!, fill!, free, to_host
include("param.jl");	export Param, update, setparam!
include("net.jl");	export Layer, LossLayer, Net, train, predict, forw, back, loss

# TODO: should work with cpu/gpu 2D/4D/5D/ND Float32/Float64
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

# TODO: fix file i/o
# include("layer.jl");	export Layer, newnet
# include("func.jl");     export relu, drop, softmaxloss, logp, logploss
# include("cuda.jl");	# extends copy!, mul!, badd!, bmul!, bsub!, sum!, zeros, rand!, fill!, free, to_host
# include("h5io.jl");	export h5write
# include("jldio.jl");    # extends JLD.save, newnet
#########################

end # module
