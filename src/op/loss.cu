#include "knet.h"

__global__ void _fill32(int n, float x, float *a) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    a[i] = x;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _fill64(int n, double x, double *a) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    a[i] = x;
    i += blockDim.x * gridDim.x;
  }
}

#include <stdint.h>

__global__ void _fill32i(int n, int32_t x, int32_t *a) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    a[i] = x;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _fill64i(int n, int64_t x, int64_t *a) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    a[i] = x;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void fill32(int n, float x, float *a) KCALL(_fill32,n,x,a);
  void fill64(int n, double x, double *a) KCALL(_fill64,n,x,a);
  void fill32i(int n, int32_t x, int32_t *a) KCALL(_fill32i,n,x,a);
  void fill64i(int n, int64_t x, int64_t *a) KCALL(_fill64i,n,x,a);
}

template<typename dType>
__global__ void _softloss(int m, int n, const dType *y, const dType *dy, const bool *mask, dType *ly) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int mn = m*n;
  while (i < mn) {
    if (mask != NULL && !mask[i/m]) {
      ly[i] = 0;
    } else {
      ly[i] = -dy[i]*log(y[i]);
    }
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softloss32(int m, int n, const float  *y, const float  *dy, const bool *mask, float  *ly) KCALL(_softloss,m,n,y,dy,mask,ly);
  void softloss64(int m, int n, const double *y, const double *dy, const bool *mask, double *ly) KCALL(_softloss,m,n,y,dy,mask,ly);
}

template<typename dType>
__global__ void _softloss_csc(int nrows, int ncols, const dType *y, int nnz, const dType *cscVal, 
			      const int *cscRowInd, const int *cscColPtr, const bool *mask, dType *ly) {
  int nz = threadIdx.x + blockIdx.x * blockDim.x;
  int col = 0;
  while (nz < nnz) {
    for (; nz > cscColPtr[col+1]-2; col++);
    if (mask != NULL && !mask[col]) {
      ly[nz] = 0;
    } else {
      dType dyi = cscVal[nz];
      int row = cscRowInd[nz]-1;
      int i = col * nrows + row;
      ly[nz] = -dyi * log(y[i]);
    }
    nz += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softloss32csc(int nrows, int ncols, const float  *y, int nnz, const float  *cscVal, const int *cscRowInd, const int *cscColPtr, const bool *mask, float  *ly) KCALL(_softloss_csc, nrows, ncols, y, nnz, cscVal, cscRowInd, cscColPtr, mask, ly);
  void softloss64csc(int nrows, int ncols, const double *y, int nnz, const double *cscVal, const int *cscRowInd, const int *cscColPtr, const bool *mask, double *ly) KCALL(_softloss_csc, nrows, ncols, y, nnz, cscVal, cscRowInd, cscColPtr, mask, ly);
}

template<typename dType>
__global__ void _softlossback(int m, int n, const dType *y, const dType *dy, const bool *mask, dType *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int mn = m*n;
  while (i < mn) {
    dx[i] = (mask != NULL && !mask[i/m]) ? 0 : (y[i] - dy[i])/(y[i] * n);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softlossback32(int m, int n, const float  *y, const float  *dy, const bool *mask, float  *dx) KCALL(_softlossback,m,n,y,dy,mask,dx);
  void softlossback64(int m, int n, const double *y, const double *dy, const bool *mask, double *dx) KCALL(_softlossback,m,n,y,dy,mask,dx);
}

template<typename dType>
__global__ void _softlossback_csc2(int nrows, int ncols, const dType *y, int nnz, const dType *cscVal, const int *cscRowInd, const int *cscColPtr, const bool *mask, dType *dx) {
  int nz = threadIdx.x + blockIdx.x * blockDim.x;
  int col = 0;
  while (nz < nnz) {
    for (; nz > cscColPtr[col+1]-2; col++);
    if (mask == NULL || mask[col]) {
      dType dy = cscVal[nz];
      int row = cscRowInd[nz]-1;
      int i = col * nrows + row;
      dx[i] *= (1 - dy/y[i]);
    }
    nz += blockDim.x * gridDim.x;
  }
}

template<typename dType>
__global__ void _softlossback_csc1(int m, int n, const bool *mask, dType *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int mn = m*n;
  dType scale = 1.0/n;
  while (i < mn) {
    dx[i] = (mask != NULL && !mask[i/m]) ? 0 : scale;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softlossback32csc(int nrows, int ncols, const float  *y, int nnz, const float  *cscVal, 
			 const int *cscRowInd, const int *cscColPtr, const bool *mask, float  *dx) {
    KCALL(_softlossback_csc1, nrows, ncols, mask, dx);
    KCALL(_softlossback_csc2, nrows, ncols, y, nnz, cscVal, cscRowInd, cscColPtr, mask, dx);
  }

  void softlossback64csc(int nrows, int ncols, const double *y, int nnz, const double *cscVal, 
			 const int *cscRowInd, const int *cscColPtr, const bool *mask, double *dx) {
    KCALL(_softlossback_csc1, nrows, ncols, mask, dx);
    KCALL(_softlossback_csc2, nrows, ncols, y, nnz, cscVal, cscRowInd, cscColPtr, mask, dx);
  }
}

__global__ void _logplossback32(int n, double scale, float *y, float *dy, float *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dx[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _logplossback64(int n, double scale, double *y, double *dy, double *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dx[i] = scale*(exp(y[i]) - dy[i]);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void logplossback32(int n, double s, float *y, float *dy, float *dx) KCALL(_logplossback32,n,s,y,dy,dx);
  void logplossback64(int n, double s, double *y, double *dy, double *dx) KCALL(_logplossback64,n,s,y,dy,dx);
}

__global__ void _xentlossback32(int nd, int nx, float *y, float *p, float *dx) {
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
}

__global__ void _xentlossback64(int nd, int nx, double *y, double *p, double *dx) {
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
}

extern "C" {
  void xentlossback32(int nd, int nx, float *y, float *p, float *dx) KCALL(_xentlossback32,nd,nx,y,p,dx);
  void xentlossback64(int nd, int nx, double *y, double *p, double *dx) KCALL(_xentlossback64,nd,nx,y,p,dx);
}

__global__ void _perclossback32(int nd, int nx, float *y, float *z, float *dx) {
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

__global__ void _perclossback64(int nd, int nx, double *y, double *z, double *dx) {
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
  void perclossback32(int nd, int nx, float *y, float *z, float *dx) KCALL(_perclossback32,nd,nx,y,z,dx);
  void perclossback64(int nd, int nx, double *y, double *z, double *dx) KCALL(_perclossback64,nd,nx,y,z,dx);
}
