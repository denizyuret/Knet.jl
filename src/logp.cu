#include "kunet.h"

__global__ void _logpforw(int nrows, int ncols, float *y) {
  /* y is layer output, i.e. unnormalized log probabilities.
     On output y will contain normalized probabilities.
  */
  float ymax, z, logz;
  int i0, i1;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  while (col < ncols) {
    i0 = col * nrows;
    i1 = i0  + nrows;
    ymax = -INFINITY;
    for (int i=i0; i<i1; i++) {
      if (y[i] > ymax) {
	ymax = y[i];
      }
    }
    z = 0;
    for (int i=i0; i<i1; i++) {
      y[i] -= ymax;
      z += exp(y[i]);
    }
    logz = log(z);
    for (int i=i0; i<i1; i++) {
      y[i] -= logz;
    }
    col += blockDim.x * gridDim.x;
  }
}

extern "C" {
void logpforw(int nrows, int ncols, float *y) KCALL(_logpforw,nrows,ncols,y);
}
