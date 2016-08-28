include("cuda1.jl")

function cuda1src(f, j=f, ex="$f(xi)"; BLK=256, THR=256)
"""
__global__ void _$(f)_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = $ex;
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
    double xi = x[i];
    y[i] = $ex;
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

for a in cuda1
    isa(a,Tuple) || (a=(a,))
    print(cuda1src(a...))
end
