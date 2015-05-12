#include "kunet.h"

__global__ void _badd(int nrows, int ncols, float *y, float *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int n = nrows * ncols;
  while (i < n) {
    y[i] += b[i % nrows];
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void badd(int nrows, int ncols, float *y, float *b) KCALL(_badd,nrows,ncols,y,b);
}
