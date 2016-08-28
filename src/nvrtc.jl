using CUDArt

const libnvrtc = Libdl.find_library(["libnvrtc"])

const saxpy = """
extern "C" __global__ 
void saxpy(float a, float *x, float *y, float *out, size_t n) 
{ 
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid < n) { 
    out[tid] = a * x[tid] + y[tid]; 
  } 
}
"""

typealias Cptr Ptr{Void}

progP = Cptr[1]

ccall((:nvrtcCreateProgram,libnvrtc),Cint,(Ptr{Cptr},Ptr{Cchar},Ptr{Cchar},Cint,Cptr,Cptr),progP,saxpy,"saxpy.cu",Cint(0),C_NULL,C_NULL)

prog = progP[1]

ccall((:nvrtcCompileProgram,libnvrtc),Cint,(nvrtcProgram,Cint,Cptr),prog,0,C_NULL)

ptxSizeP = Csize_t[1]

ccall((:nvrtcGetPTXSize,libnvrtc),Cint,(nvrtcProgram,Ptr{Csize_t}),prog,ptxSizeP)

ptxSize = ptxSizeP[1]
ptx = Array(Cchar,ptxSize)

ccall((:nvrtcGetPTX,libnvrtc),Cint,(nvrtcProgram,Ptr{Cchar}),prog,ptx)

ccall((:nvrtcDestroyProgram,libnvrtc),Cint,(Ptr{nvrtcProgram},),progP)

# Check out the PTX example in CUDArt to see how this is done:

# CUdevice cuDevice;
# CUcontext context;
# CUmodule module;
# CUfunction kernel;
# cuInit(0);
# cuDeviceGet(&cuDevice, 0);
# cuCtxCreate(&context, 0, cuDevice);
# cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
# cuModuleGetFunction(&kernel, module, "saxpy");

n = 10
a = rand()
x = KnetArray(rand(n))
y = KnetArray(rand(n))
out = similar(x)

# cuLaunchKernel(kernel, 
#                NUM_THREADS, 1, 1, // grid dim
#                NUM_BLOCKS, 1, 1, // block dim 
#                0, NULL, // shared mem and stream 
#                args, // arguments 
#                0);

