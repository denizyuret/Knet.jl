#include "kunet.h"

__global__ void _drop32(int n, float *x, float *xmask, double dropout, double scale) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (xmask[i] < dropout) x[i] = 0;
    else x[i] *= scale;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _drop64(int n, double *x, double *xmask, double dropout, double scale) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (xmask[i] < dropout) x[i] = 0;
    else x[i] *= scale;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
void drop32(int n, float *x, float *xmask, double dropout, double scale) KCALL(_drop32,n,x,xmask,dropout,scale);
void drop64(int n, double *x, double *xmask, double dropout, double scale) KCALL(_drop64,n,x,xmask,dropout,scale);
}
