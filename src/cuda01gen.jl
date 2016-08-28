using Knet: cuda01

function cuda01src(f, j=f, ex="$f(s,xi)"; BLK=256, THR=256)
"""
__global__ void _$(f)_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $(f)_32_01(int n, float s, float *x, float *y) {
    _$(f)_32_01<<<$BLK,$THR>>>(n,s,x,y);
  }    
}
__global__ void _$(f)_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $(f)_64_01(int n, double s, double *x, double *y) {
    _$(f)_64_01<<<$BLK,$THR>>>(n,s,x,y);
  }    
}
"""
end

for a in cuda01
    isa(a,Tuple) || (a=(a,))
    print(cuda01src(a...))
end
