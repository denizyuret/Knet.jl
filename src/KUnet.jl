module KUnet
const libkunet = find_library(["libkunet"], [Pkg.dir("KUnet/cuda")])

using InplaceOps
using Base.LinAlg.BLAS
using HDF5

# Conditional module import
installed(pkg)=isdir(Pkg.dir(string(pkg)))
macro useif(pkg) if installed(pkg) Expr(:using,pkg) end end
@useif CUDArt
@useif CUBLAS

# Turn gpu support on/off using KUnet.gpu(true/false)
function gpu(b::Bool)
    global usegpu = b
    usegpu == false     && return
    libkunet == ""      && (warn("libkunet.so not found."); usegpu=false)
    !isdefined(:CUDArt) && (warn("CUDArt not installed."); usegpu=false)
    !isdefined(:CUBLAS) && (warn("CUBLAS not installed."); usegpu=false)
end

global usegpu
gpu((libkunet != "") && isdefined(:CUDArt) && isdefined(:CUBLAS))

#########################
include("types.jl");	export Layer, Net, UpdateParam, setparam!
include("cuda.jl");	# extends copy!, mul!, badd!, bmul!, bsub!, sum!, zeros, rand!, fill!, free, to_host
include("net.jl");	export train, predict
include("update.jl");	# implements update: helper for train
include("func.jl");     export relu, drop, softmaxloss
include("h5io.jl");	export h5write
#########################

end # module
