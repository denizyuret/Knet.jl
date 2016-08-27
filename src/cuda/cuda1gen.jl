include("cuda1arg.jl")

function cuda1src(f, j=f, ex32="$(f)(x[i])", ex64=ex32; BLK=128, THR=128)
"""
__global__ void _$(f)_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = $ex32;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $(f)_32(int n, float *x, float *y) {
    _$(f)_32<<<$BLK,$THR>>>(n,x,y);
  }    
}
__global__ void _$(f)_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = $ex64;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $(f)_64(int n, double *x, double *y) {
    _$(f)_64<<<$BLK,$THR>>>(n,x,y);
  }    
}
"""
end

for a in cuda1arg
    isa(a,Tuple) || (a=(a,))
    print(cuda1src(a...))
end
