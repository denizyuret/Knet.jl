#include "kunet.h"

__global__ void _sxentloss(int nd, int nx, float *y, float *dy) {
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

__global__ void _dxentloss(int nd, int nx, double *y, double *dy) {
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
  void sxentloss(int nd, int nx, float *y, float *dy) KCALL(_sxentloss,nd,nx,y,dy);
  void dxentloss(int nd, int nx, double *y, double *dy) KCALL(_dxentloss,nd,nx,y,dy);
}
