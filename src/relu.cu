#include "kunet.h"

__global__ void _reluforw(int n, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] < 0) y[i] = 0;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _reluback(int n, float *y, float *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] <= 0) dy[i] = 0;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
void reluforw(int n, float *y) KCALL(_reluforw,n,y);
void reluback(int n, float *y, float *dy) KCALL(_reluback,n,y,dy);
}
