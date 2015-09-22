# See if we have gpu support.  This determines whether gpu code is
# loaded, not whether it is used.  The user can control gpu use by
# using the gpu() function.
GPU = true
lpath = [Pkg.dir("KUnet/src")]
for l in ("libkunet", "libcuda", "libcudart", "libcublas", "libcudnn")
    isempty(Libdl.find_library([l], lpath)) && (warn("Cannot find $l");GPU=false)
end
for p in ("CUDArt", "CUBLAS", "CUDNN")
    isdir(Pkg.dir(p)) || (warn("Cannot find $p");GPU=false)
end
if GPU
    gpucnt=Int32[0]
    gpuret=ccall((:cudaGetDeviceCount,"libcudart"),Int32,(Ptr{Cint},),gpucnt)
    (gpucnt == 0 || gpuret != 0) && (warn("No gpu detected");GPU=false)
end

# GPU is a variable indicating the existence of a gpu.
GPU || warn("Using the cpu")

# USEGPU and its controlling function gpu() allows the user 
# to control whether the gpu will be used.
USEGPU = GPU
gpu()=USEGPU
gpu(b::Bool)=(b && !GPU && error("No GPU"); global USEGPU=b)

# Conditionally import gpulibs
macro useifgpu(pkg) if GPU Expr(:using,pkg) end end

# Conditionally evaluate expressions
macro gpu(_ex); if GPU; esc(_ex); end; end

# Load kernels from CUDArt
@useifgpu CUDArt
GPU && CUDArt.init!([CUDArt.CuModule(),], [CUDArt.device(),])

# Additional cuda code
const libkunet = Libdl.find_library(["libkunet"], [Pkg.dir("KUnet/src")])

# For debugging
function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart.so"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    convert(Int,mfree[1])
end

# This is for profiling:
gpusync()=device_synchronize()
# This is for production:
# gpusync()=nothing

# This gets overriden if gpu available:
setseed(n)=srand(n)
