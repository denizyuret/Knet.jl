#include "knet.h"

template<typename dType>
__global__ void _axpb(int n, dType a, dType *x, dType p, dType b, dType *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dType yi = x[i];
    if (p != 1) yi = pow(yi,p);
    if (a != 1) yi *= a;
    if (b != 0) yi += b;
    y[i] = yi;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void axpb32(int n, float a, float *x, float p, float b, float *y) KCALL(_axpb,n,a,x,p,b,y)
  void axpb64(int n, double a, double *x, double p, double b, double *y) KCALL(_axpb,n,a,x,p,b,y)
}

/* x is layer output, i.e. unnormalized log probabilities.
   On output y will contain normalized log probabilities. 
   x and y can point to the same array. */

template<typename dType>
__global__ void _logpforw(int nrows, int ncols, dType *x, dType *y) {
  double expy;
  dType xmax, logz;
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
  void logpforw32(int nrows, int ncols, float *x, float *y) KCALL(_logpforw,nrows,ncols,x,y);
  void logpforw64(int nrows, int ncols, double *x, double *y) KCALL(_logpforw,nrows,ncols,x,y);
}
