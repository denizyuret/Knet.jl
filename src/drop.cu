#include "kunet.h"

__global__ void _sdrop(int n, float *x, float *xmask, float dropout, float scale) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (xmask[i] < dropout) x[i] = 0;
    else x[i] *= scale;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _ddrop(int n, double *x, double *xmask, double dropout, double scale) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (xmask[i] < dropout) x[i] = 0;
    else x[i] *= scale;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
void sdrop(int n, float *x, float *xmask, float dropout, float scale) KCALL(_sdrop,n,x,xmask,dropout,scale);
void ddrop(int n, double *x, double *xmask, double dropout, double scale) KCALL(_ddrop,n,x,xmask,dropout,scale);
}
