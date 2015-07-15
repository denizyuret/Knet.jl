#include "kunet.h"

__global__ void _axpb32(int n, float a, float b, float *x) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[i] = a*x[i]+b;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _axpb64(int n, double a, double b, double *x) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[i] = a*x[i]+b;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void axpb32(int n, float a, float b, float *x) KCALL(_axpb32, n, a, b, x);
  void axpb64(int n, double a, double b, double *x) KCALL(_axpb64, n, a, b, x);
}
