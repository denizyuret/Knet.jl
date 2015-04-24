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


#########################
include("types.jl");	export Layer, Net, UpdateParam, setparam!, newnet
include("cuda.jl");	# extends copy!, mul!, badd!, bmul!, bsub!, sum!, zeros, rand!, fill!, free, to_host
include("net.jl");	export train, predict
include("update.jl");	# implements update: helper for train
include("func.jl");     export relu, drop, softmaxloss, logp, logploss
include("h5io.jl");	export h5write
include("jldio.jl");    # extends JLD.save, newnet
#########################

end # module
