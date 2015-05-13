module KUnet
using Compat
using InplaceOps
using Base.LinAlg.BLAS
using HDF5, JLD

# GPU support is on by default if the required libraries exist.
# The user can turn gpu support on/off using KUnet.gpu(true/false)

global usegpu
const libkunet = find_library(["libkunet"], [Pkg.dir("KUnet/cuda")])
const libcuda = find_library(["libcuda"])
const libcudart = find_library(["libcudart", "cudart"])
installed(pkg)=isdir(Pkg.dir(string(pkg)))

function gpu(b::Bool)
    global usegpu = b
    usegpu == false     && return
    isempty(libkunet)   && (warn("libkunet not found."); usegpu=false)
    isempty(libcuda)    && (warn("libcuda not found."); usegpu=false)
    isempty(libcudart)  && (warn("libcudart not found."); usegpu=false)
    !installed(:CUDArt) && (warn("CUDArt not installed."); usegpu=false)
    !installed(:CUBLAS) && (warn("CUBLAS not installed."); usegpu=false)
    usegpu
end

gpu(true)

# Conditional module import
macro useifgpu(pkg) if usegpu Expr(:using,pkg) end end
@useifgpu CUDArt
@useifgpu CUBLAS
# TODO: currently conv layers only have gpu impl based on cudnn, we
# need a cpu implementation and we need to make it generic so the same
# code works whether or not cudnn / gpu is available.
@useifgpu CUDNN  

#########################
# TODO: clean util.jl, minimize cuda code
include("util.jl");	# extends copy!, mul!, badd!, bmul!, bsub!, sum!, zeros, rand!, fill!, free, to_host
include("param.jl");	export Param, update, setparam!
include("loss.jl");	export softmaxloss, logploss
include("net.jl");	export Layer, Net, train, predict, forw, back

# TODO: should work with cpu/gpu 2D/4D/5D/ND Float32/Float64
include("mmul.jl");     export Mmul
include("bias.jl");	export Bias
include("conv.jl");	export Conv
include("pool.jl");	export Pool
include("relu.jl");	export Relu
include("tanh.jl");	export Tanh
include("sigm.jl");	export Sigm
include("drop.jl");	export Drop
include("logp.jl");	export Logp

# TODO: fix file i/o
# include("layer.jl");	export Layer, newnet
# include("func.jl");     export relu, drop, softmaxloss, logp, logploss
# include("cuda.jl");	# extends copy!, mul!, badd!, bmul!, bsub!, sum!, zeros, rand!, fill!, free, to_host
# include("h5io.jl");	export h5write
# include("jldio.jl");    # extends JLD.save, newnet
#########################

end # module
