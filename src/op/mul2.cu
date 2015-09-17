#include "kunet.h"

__global__ void _emul32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] *= x[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _emul64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] *= x[i];
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void emul32(int n, float  *x, float  *y) KCALL(_emul32,n,x,y);
  void emul64(int n, double *x, double *y) KCALL(_emul64,n,x,y);
}
