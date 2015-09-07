#include "kunet.h"

__global__ void _softloss32(int n, double scale, float *y, float *dy, float *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dx[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _softloss64(int n, double scale, double *y, double *dy, double *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dx[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softloss32(int n, double s, float *y, float *dy, float *dx) KCALL(_softloss32,n,s,y,dy,dx);
  void softloss64(int n, double s, double *y, double *dy, double *dx) KCALL(_softloss64,n,s,y,dy,dx);
}

__global__ void _logploss32(int n, double scale, float *y, float *dy, float *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dx[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _logploss64(int n, double scale, double *y, double *dy, double *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dx[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void logploss32(int n, double s, float *y, float *dy, float *dx) KCALL(_logploss32,n,s,y,dy,dx);
  void logploss64(int n, double s, double *y, double *dy, float *dx) KCALL(_logploss64,n,s,y,dy,dx);
}

__global__ void _xentloss32(int nd, int nx, float *y, float *p, float *dx) {
  double z;
  float qi, ymax;
  int i0, i1;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  while (ix < nx) {
    i0 = ix * nd;
    i1 = i0 + nd;
    z = 0;
    ymax = -INFINITY;
    for (int i=i0; i<i1; i++) { if (y[i] > ymax) ymax = y[i]; }
    for (int i=i0; i<i1; i++) { z += exp(y[i] - ymax); }
    for (int i=i0; i<i1; i++) { qi = exp(y[i] - ymax)/z; dx[i] = (qi - p[i])/nx; }
    ix += blockDim.x * gridDim.x;
  }
  // free(qz);
}

__global__ void _xentloss64(int nd, int nx, double *y, double *p, double *dx) {
  double z;
  double qi, ymax;
  int i0, i1;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  while (ix < nx) {
    i0 = ix * nd;
    i1 = i0 + nd;
    z = 0;
    ymax = -INFINITY;
    for (int i=i0; i<i1; i++) { if (y[i] > ymax) ymax = y[i]; }
    for (int i=i0; i<i1; i++) { z += exp(y[i] - ymax); }
    for (int i=i0; i<i1; i++) { qi = exp(y[i] - ymax)/z; dx[i] = (qi - p[i])/nx; }
    ix += blockDim.x * gridDim.x;
  }
  // free(qz);
}


extern "C" {
  void xentloss32(int nd, int nx, float *y, float *p, float *dx) KCALL(_xentloss32,nd,nx,y,p,dx);
  void xentloss64(int nd, int nx, double *y, double *p, double *dx) KCALL(_xentloss64,nd,nx,y,p,dx);
}

__global__ void _percloss32(int nd, int nx, float *y, float *z, float *dx) {
  float ymax, zmax;
  int i0, i1, cy, cz;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  while (ix < nx) {
    ymax = -INFINITY; cy = 0;
    zmax = -INFINITY; cz = 0;
    i0 = ix * nd; i1 = i0 + nd;
    for (int i=i0; i<i1; i++) { 
      if (y[i] > ymax) { ymax = y[i]; cy = i; } 
      if (z[i] > zmax) { zmax = z[i]; cz = i; } 
      dx[i] = 0;
    }
    if (cz != cy) {
      dx[cz] = -1;
      dx[cy] = 1;
    }
    ix += blockDim.x * gridDim.x;
  }
}

__global__ void _percloss64(int nd, int nx, double *y, double *z, double *dx) {
  double ymax, zmax;
  int i0, i1, cy, cz;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  while (ix < nx) {
    i0 = ix * nd; i1 = i0 + nd;
    ymax = -INFINITY; cy = 0;
    zmax = -INFINITY; cz = 0;
    for (int i=i0; i<i1; i++) { 
      if (y[i] > ymax) { ymax = y[i]; cy = i; } 
      if (z[i] > zmax) { zmax = z[i]; cz = i; } 
      dx[i] = 0;
    }
    if (cz != cy) {
      dx[cz] = -1;
      dx[cy] = 1;
    }
    ix += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void percloss32(int nd, int nx, float *y, float *z, float *dx) KCALL(_percloss32,nd,nx,y,z,dx);
  void percloss64(int nd, int nx, double *y, double *z, double *dx) KCALL(_percloss64,nd,nx,y,z,dx);
}
