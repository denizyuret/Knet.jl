#include "kunet.h"

/* x is layer output, i.e. unnormalized log probabilities.
   On output y will contain normalized log probabilities. 
   x and y can point to the same array. */

__global__ void _logpforw32(int nrows, int ncols, float *x, float *y) {
  double expy;
  float xmax, logz;
  int i0, i1;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  while (col < ncols) {
    i0 = col * nrows;
    i1 = i0  + nrows;
    xmax = -INFINITY;
    for (int i=i0; i<i1; i++) { if (x[i] > xmax) xmax = x[i]; }
    expy = 0;
    for (int i=i0; i<i1; i++) { y[i] = x[i]-xmax; expy += exp(y[i]); }
    logz = log(expy);
    for (int i=i0; i<i1; i++) { y[i] -= logz; }
    col += blockDim.x * gridDim.x;
  }
}

__global__ void _logpforw64(int nrows, int ncols, double *x, double *y) {
  double xmax, expy, logz;
  int i0, i1;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  while (col < ncols) {
    i0 = col * nrows;
    i1 = i0  + nrows;
    xmax = -INFINITY;
    for (int i=i0; i<i1; i++) { if (x[i] > xmax) xmax = x[i]; }
    expy = 0;
    for (int i=i0; i<i1; i++) { y[i] = x[i]-xmax; expy += exp(y[i]); }
    logz = log(expy);
    for (int i=i0; i<i1; i++) { y[i] -= logz; }
    col += blockDim.x * gridDim.x;
  }
}

extern "C" {
void logpforw32(int nrows, int ncols, float *x, float *y) KCALL(_logpforw32,nrows,ncols,x,y);
void logpforw64(int nrows, int ncols, double *x, double *y) KCALL(_logpforw64,nrows,ncols,x,y);
}
