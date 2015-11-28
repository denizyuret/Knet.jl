#include "knet.h"

/* y is layer output, i.e. unnormalized log probabilities.
   On output y will contain normalized probabilities. */

__global__ void _logpforw32(int nrows, int ncols, float *y) {
  double z;
  float ymax, logz;
  int i0, i1;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  while (col < ncols) {
    i0 = col * nrows;
    i1 = i0  + nrows;
    ymax = -INFINITY;
    for (int i=i0; i<i1; i++) { if (y[i] > ymax) ymax = y[i]; }
    z = 0;
    for (int i=i0; i<i1; i++) { y[i] -= ymax; z += exp(y[i]); }
    logz = log(z);
    for (int i=i0; i<i1; i++) { y[i] -= logz; }
    col += blockDim.x * gridDim.x;
  }
}

__global__ void _logpforw64(int nrows, int ncols, double *y) {
  double ymax, z, logz;
  int i0, i1;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  while (col < ncols) {
    i0 = col * nrows;
    i1 = i0  + nrows;
    ymax = -INFINITY;
    for (int i=i0; i<i1; i++) { if (y[i] > ymax) ymax = y[i]; }
    z = 0;
    for (int i=i0; i<i1; i++) { y[i] -= ymax; z += exp(y[i]); }
    logz = log(z);
    for (int i=i0; i<i1; i++) { y[i] -= logz; }
    col += blockDim.x * gridDim.x;
  }
}

extern "C" {
void logpforw32(int nrows, int ncols, float *y) KCALL(_logpforw32,nrows,ncols,y);
void logpforw64(int nrows, int ncols, double *y) KCALL(_logpforw64,nrows,ncols,y);
}
