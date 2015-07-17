#include "kunet.h"

__global__ void _adagrad32(int n, float eps, float *dw2, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dw2[i] += dw[i] * dw[i];
    dw[i] /= (eps + sqrt(dw2[i]));
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _l1reg32(int n, float l1, float *w, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (w[i] > 0) dw[i] += l1;
    else if (w[i] < 0) dw[i] -= l1;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _adagrad64(int n, double eps, double *dw2, double *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dw2[i] += dw[i] * dw[i];
    dw[i] /= (eps + sqrt(dw2[i]));
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _l1reg64(int n, double l1, double *w, double *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (w[i] > 0) dw[i] += l1;
    else if (w[i] < 0) dw[i] -= l1;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void l1reg32(int n, float l1, float *w, float *dw) KCALL(_l1reg32,n,l1,w,dw);
  void l1reg64(int n, double l1, double *w, double *dw) KCALL(_l1reg64,n,l1,w,dw);
  void adagrad32(int n, float eps, float *dw2, float *dw) KCALL(_adagrad32,n,eps,dw2,dw);
  void adagrad64(int n, double eps, double *dw2, double *dw) KCALL(_adagrad64,n,eps,dw2,dw);
}
