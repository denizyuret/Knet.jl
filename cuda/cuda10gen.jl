include("cuda10.jl")

function cuda10src(f, j, ex; BLK=256, THR=256)
"""
__global__ void _$(f)_32_10(int n, float *x, float s, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $(f)_32_10(int n, float *x, float s, float *y) {
    _$(f)_32_10<<<$BLK,$THR>>>(n,x,s,y);
  }    
}
__global__ void _$(f)_64_10(int n, double *x, double s, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $(f)_64_10(int n, double *x, double s, double *y) {
    _$(f)_64_10<<<$BLK,$THR>>>(n,x,s,y);
  }    
}
"""
end

for a in cuda10
    isa(a,Tuple) || (a=(a,))
    print(cuda10src(a...))
end
