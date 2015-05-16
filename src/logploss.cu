#include "kunet.h"

__global__ void _slogploss(int n, float scale, float *y, float *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _dlogploss(int n, double scale, double *y, double *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void slogploss(int n, float s, float *y, float *dy) KCALL(_slogploss,n,s,y,dy);
  void dlogploss(int n, double s, double *y, double *dy) KCALL(_dlogploss,n,s,y,dy);
}
