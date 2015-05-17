#include "kunet.h"

__global__ void _softloss32(int n, float scale, float *y, float *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _softloss64(int n, double scale, double *y, double *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softloss32(int n, float s, float *y, float *dy) KCALL(_softloss32,n,s,y,dy);
  void softloss64(int n, double s, double *y, double *dy) KCALL(_softloss64,n,s,y,dy);
}
