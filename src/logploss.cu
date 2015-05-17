#include "kunet.h"

__global__ void _logploss32(int n, float scale, float *y, float *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _logploss64(int n, double scale, double *y, double *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void logploss32(int n, float s, float *y, float *dy) KCALL(_logploss32,n,s,y,dy);
  void logploss64(int n, double s, double *y, double *dy) KCALL(_logploss64,n,s,y,dy);
}
