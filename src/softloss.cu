#include "kunet.h"

__global__ void _ssoftloss(int n, float scale, float *y, float *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _dsoftloss(int n, double scale, double *y, double *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void ssoftloss(int n, float s, float *y, float *dy) KCALL(_ssoftloss,n,s,y,dy);
  void dsoftloss(int n, double s, double *y, double *dy) KCALL(_dsoftloss,n,s,y,dy);
}
