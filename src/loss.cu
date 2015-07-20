#include "kunet.h"

__global__ void _softloss32(int n, double scale, float *y, float *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _softloss64(int n, double scale, double *y, double *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softloss32(int n, double s, float *y, float *dy) KCALL(_softloss32,n,s,y,dy);
  void softloss64(int n, double s, double *y, double *dy) KCALL(_softloss64,n,s,y,dy);
}

__global__ void _logploss32(int n, double scale, float *y, float *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _logploss64(int n, double scale, double *y, double *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dy[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void logploss32(int n, double s, float *y, float *dy) KCALL(_logploss32,n,s,y,dy);
  void logploss64(int n, double s, double *y, double *dy) KCALL(_logploss64,n,s,y,dy);
}

__global__ void _xentloss32(int nd, int nx, float *y, float *dy) {
  double z, ymax;
  // double *qz = (double *) malloc(nd * sizeof(double));
  int i0, i1;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  while (ix < nx) {
    i0 = ix * nd;
    i1 = i0 + nd;
    z = 0;
    ymax = -INFINITY;
    for (int i=i0; i<i1; i++) { if (y[i] > ymax) ymax = y[i]; }
    for (int i=i0; i<i1; i++) { y[i] = exp(y[i] - ymax); z+=y[i]; }
    for (int i=i0; i<i1; i++) { y[i] /= z; dy[i] = (y[i] - dy[i])/nx; }
    //for (int i=i0; i<i1; i++) { z += (qz[i-i0] = exp(y[i] - ymax)); }
    //for (int i=i0; i<i1; i++) { dy[i] = (qz[i-i0]/z - dy[i])/nx; }
    ix += blockDim.x * gridDim.x;
  }
  // free(qz);
}

__global__ void _xentloss64(int nd, int nx, double *y, double *dy) {
  double z, ymax;
  // double *qz = (double *) malloc(nd * sizeof(double));
  int i0, i1;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  while (ix < nx) {
    i0 = ix * nd;
    i1 = i0 + nd;
    z = 0;
    ymax = -INFINITY;
    for (int i=i0; i<i1; i++) { if (y[i] > ymax) ymax = y[i]; }
    for (int i=i0; i<i1; i++) { y[i] = exp(y[i] - ymax); z+=y[i]; }
    for (int i=i0; i<i1; i++) { y[i] /= z; dy[i] = (y[i] - dy[i])/nx; }
    // for (int i=i0; i<i1; i++) { z += (qz[i-i0] = exp(y[i] - ymax)); }
    // for (int i=i0; i<i1; i++) { dy[i] = (qz[i-i0]/z - dy[i])/nx; }
    ix += blockDim.x * gridDim.x;
  }
  // free(qz);
}

extern "C" {
  void xentloss32(int nd, int nx, float *y, float *dy) KCALL(_xentloss32,nd,nx,y,dy);
  void xentloss64(int nd, int nx, double *y, double *dy) KCALL(_xentloss64,nd,nx,y,dy);
}
