# Macro to conditionally import gpulibs
macro useifgpu(pkg) if GPU Expr(:using,pkg) end end

# Macro to conditionally evaluate expressions
macro gpu(_ex); if GPU; esc(_ex); end; end

# GPU is a variable that shows whether the system has gpu support
# See if we have gpu support.  This determines whether gpu code is
# loaded, not whether it is used.  The user can control gpu use by
# using the gpu() function.
GPU = true
lpath = [Pkg.dir("Knet/src")]
for l in ("libknet", "libcuda", "libcudart", "libcublas", "libcudnn")
    isempty(Libdl.find_library([l], lpath)) && (warn("Cannot find $l");GPU=false)
end
for p in ("CUDArt", "CUBLAS", "CUDNN")
    isdir(Pkg.dir(p)) || (warn("Cannot find $p");GPU=false)
end
if GPU
    gpucnt=Int32[0]
    @gpu gpuret=ccall((:cudaGetDeviceCount,"libcudart"),Int32,(Ptr{Cint},),gpucnt)
    (gpucnt == 0 || gpuret != 0) && (warn("No gpu detected");GPU=false)
end

# GPU is a variable indicating the existence of a gpu.
GPU || warn("Using the cpu")

# USEGPU and its controlling function gpu() allows the user 
# to control whether the gpu will be used.
USEGPU = GPU
gpu()=USEGPU
gpu(b::Bool)=(b && !GPU && error("No GPU"); global USEGPU=b)

# Additional cuda code from Knet
const libknet = Libdl.find_library(["libknet"], [Pkg.dir("Knet/src")])

# For debugging
@gpu function gpumem()
    mfree=Csize_t[1]
    mtotal=Csize_t[1]
    ccall((:cudaMemGetInfo,"libcudart"),Cint,(Ptr{Csize_t},Ptr{Csize_t}),mfree,mtotal)
    nbytes=convert(Int,mfree[1])
    narray=length(CUDArt.cuda_ptrs)
    (narray,nbytes)
end

# setseed: Set both cpu and gpu seed. This gets overwritten in curand.jl if gpu available
setseed(n)=srand(n)

# to_host: Set this to identity unless input is CudaArray.
@useifgpu CUDArt
if GPU
    CUDArt.to_host(x)=x
else
    to_host(x)=x
end
