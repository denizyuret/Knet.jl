#include "kunet.h"

__global__ void _percloss32(int nd, int nx, float *y, float *z, float *dy) {
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
      dy[i] = 0;
    }
    if (cz != cy) {
      dy[cz] = -1;
      dy[cy] = 1;
    }
    ix += blockDim.x * gridDim.x;
  }
}

__global__ void _percloss64(int nd, int nx, double *y, double *z, double *dy) {
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
      dy[i] = 0;
    }
    if (cz != cy) {
      dy[cz] = -1;
      dy[cy] = 1;
    }
    ix += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void percloss32(int nd, int nx, float *y, float *z, float *dy) KCALL(_percloss32,nd,nx,y,z,dy);
  void percloss64(int nd, int nx, double *y, double *z, double *dy) KCALL(_percloss64,nd,nx,y,z,dy);
}
