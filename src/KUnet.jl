module KUnet

using InplaceOps
using Base.LinAlg.BLAS
using HDF5

# Conditional module import
installed(pkg)=isdir(Pkg.dir(string(pkg)))
macro useif(pkg) if installed(pkg) Expr(:using,pkg) end end

# Turn gpu support on/off using KUnet.gpu(true/false)
function gpu(b::Bool)
    global usegpu = b
    usegpu == false     && return
    libkunet == ""      && (warn("libkunet.so not found."); usegpu=false)
    !isdefined(:CUDArt) && (warn("CUDArt not installed."); usegpu=false)
    !isdefined(:CUBLAS) && (warn("CUBLAS not installed."); usegpu=false)
end
const libkunet = find_library(["libkunet"], [Pkg.dir("KUnet/cuda")])
@useif CUDArt
@useif CUBLAS
gpu(libkunet != "" && isdefined(:CUDArt) && isdefined(:CUBLAS))

###################
include("types.jl")
include("cuda.jl")
include("net.jl")
include("update.jl")
include("func.jl")
include("h5io.jl")
###################

end # module
