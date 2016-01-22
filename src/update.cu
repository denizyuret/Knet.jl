#include "knet.h"

__global__ void _adagrad32(int n, double eps, float *dw2, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dw2[i] += dw[i] * dw[i];
    dw[i] /= sqrt(dw2[i] + eps);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _l1reg32(int n, double l1, float *w, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (w[i] > 0) dw[i] += l1;
    else if (w[i] < 0) dw[i] -= l1;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _rmsprop32(int n, double eps, double rho, float *dw2, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dw2[i] = dw2[i] * rho + (1 - rho) * dw[i] * dw[i];
    dw[i] /= sqrt(dw2[i] + eps);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _adam32(int n, int t, double eps, double b1, double b2, float *fstm, float *scndm, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    fstm[i] = b1*fstm[i] + (1-b1)*dw[i];
    scndm[i] = b2*scndm[i] + (1-b2)*(dw[i] *dw[i]);
    dw[i] = (fstm[i] / (1 - pow(b1,(double)t))) / (sqrt(scndm[i] / (1 - pow(b2,(double)t))) + eps);

    i += blockDim.x * gridDim.x;
  }
}

__global__ void _adagrad64(int n, double eps, double *dw2, double *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dw2[i] += dw[i] * dw[i];
    dw[i] /= sqrt(dw2[i] + eps);
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

__global__ void _rmsprop64(int n, double eps, double rho, double *dw2, double *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dw2[i] = dw2[i] * rho + (1 - rho) * dw[i] * dw[i];
    dw[i] /= sqrt(dw2[i] + eps);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _adam64(int n, int t, double eps, double b1, double b2, double *fstm, double *scndm, double *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    fstm[i] = b1*fstm[i] + (1-b1)*dw[i];
    scndm[i] = b2*scndm[i] + (1-b2)*(dw[i] *dw[i]);
    dw[i] = (fstm[i] / (1 - pow(b1,(double)t))) / (sqrt(scndm[i] / (1 - pow(b2,(double)t))) + eps);

    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void l1reg32(int n, double l1, float *w, float *dw) KCALL(_l1reg32,n,l1,w,dw);
  void l1reg64(int n, double l1, double *w, double *dw) KCALL(_l1reg64,n,l1,w,dw);
  void adagrad32(int n, double eps, float *dw2, float *dw) KCALL(_adagrad32,n,eps,dw2,dw);
  void adagrad64(int n, double eps, double *dw2, double *dw) KCALL(_adagrad64,n,eps,dw2,dw);
  void rmsprop32(int n, double eps, double rho, float *dw2, float *dw) KCALL(_rmsprop32,n,eps, rho, dw2,dw);
  void rmsprop64(int n, double eps, double rho, double *dw2, double *dw) KCALL(_rmsprop64,n,eps, rho, dw2,dw);
  void adam32(int n, int t, double eps, double b1, double b2, float *fstm, float *scndm, float *dw) KCALL(_adam32,n, t, eps, b1, b2, fstm, scndm,dw);
  void adam64(int n, int t, double eps, double b1, double b2, double *fstm, double *scndm, double *dw) KCALL(_adam64,n, t, eps, b1, b2, fstm, scndm,dw);
}
