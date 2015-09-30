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

__global__ void _softloss32(int n, float *y, float *dy, float *ly) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    ly[i] = -dy[i]*log(y[i]);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _softloss64(int n, double *y, double *dy, double *ly) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    ly[i] = -dy[i]*log(y[i]);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softloss32(int n, float *y, float *dy, float *ly)    KCALL(_softloss32,n,y,dy,ly);
  void softloss64(int n, double *y, double *dy, double *ly) KCALL(_softloss64,n,y,dy,ly);
}

__global__ void _softloss32csc(int nrows, int ncols, float *y, const int nnz, const float *cscVal, const int *cscRowInd, const int *cscColPtr, float *ly) {
  int nz = threadIdx.x + blockIdx.x * blockDim.x;
  while (nz < nnz) {
    float dyi = cscVal[nz];
    int row = cscRowInd[nz]-1;
    int col; for (col = 0; nz > cscColPtr[col+1]-2; col++);
    int i = col * nrows + row;
    ly[nz] = -dyi * log(y[i]);
    nz += blockDim.x * gridDim.x;
  }
}

__global__ void _softloss64csc(int nrows, int ncols, double *y, const int nnz, const double *cscVal, const int *cscRowInd, const int *cscColPtr, double *ly) {
  int nz = threadIdx.x + blockIdx.x * blockDim.x;
  while (nz < nnz) {
    double dyi = cscVal[nz];
    int row = cscRowInd[nz]-1;
    int col; for (col = 0; nz > cscColPtr[col+1]-2; col++);
    int i = col * nrows + row;
    ly[nz] = -dyi * log(y[i]);
    nz += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softloss32csc(int nrows, int ncols, float  *y, const int nnz, const float  *cscVal, const int *cscRowInd, const int *cscColPtr, float  *ly) KCALL(_softloss32csc, nrows, ncols, y, nnz, cscVal, cscRowInd, cscColPtr, ly);
  void softloss64csc(int nrows, int ncols, double *y, const int nnz, const double *cscVal, const int *cscRowInd, const int *cscColPtr, double *ly) KCALL(_softloss64csc, nrows, ncols, y, nnz, cscVal, cscRowInd, cscColPtr, ly);
}

__global__ void _softlossback32(int n, double scale, float *y, float *dy, float *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dx[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _softlossback64(int n, double scale, double *y, double *dy, double *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dx[i] = scale*(y[i] - dy[i])/y[i];
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softlossback32(int n, double s, float *y, float *dy, float *dx) KCALL(_softlossback32,n,s,y,dy,dx);
  void softlossback64(int n, double s, double *y, double *dy, double *dx) KCALL(_softlossback64,n,s,y,dy,dx);
}

__global__ void _softlossback32csc(int nrows, int ncols, float *y, const int nnz, const float *cscVal, const int *cscRowInd, const int *cscColPtr, float *dx) {
  int nz = threadIdx.x + blockIdx.x * blockDim.x;
  while (nz < nnz) {
    float dy = cscVal[nz];
    int row = cscRowInd[nz]-1;
    int col; for (col = 0; nz > cscColPtr[col+1]-2; col++);
    int i = col * nrows + row;
    dx[i] *= (1 - dy/y[i]);
    nz += blockDim.x * gridDim.x;
  }
}

__global__ void _softlossback64csc(int nrows, int ncols, double *y, const int nnz, const double *cscVal, const int *cscRowInd, const int *cscColPtr, double *dx) {
  int nz = threadIdx.x + blockIdx.x * blockDim.x;
  while (nz < nnz) {
    double dy = cscVal[nz];
    int row = cscRowInd[nz]-1;
    int col; for (col = 0; nz > cscColPtr[col+1]-2; col++);
    int i = col * nrows + row;
    dx[i] *= (1 - dy/y[i]);
    nz += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void softlossback32csc(int nrows, int ncols, float  *y, const int nnz, const float  *cscVal, 
			 const int *cscRowInd, const int *cscColPtr, float  *dx) {
    KCALL(_fill32, nrows*ncols, 1.0/ncols, dx);
    KCALL(_softlossback32csc, nrows, ncols, y, nnz, cscVal, cscRowInd, cscColPtr, dx);
  }

  void softlossback64csc(int nrows, int ncols, double *y, const int nnz, const double *cscVal, 
			 const int *cscRowInd, const int *cscColPtr, double *dx) {
    KCALL(_fill64, nrows*ncols, 1.0/ncols, dx);
    KCALL(_softlossback64csc, nrows, ncols, y, nnz, cscVal, cscRowInd, cscColPtr, dx);
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
