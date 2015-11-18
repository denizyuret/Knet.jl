#include "../knet.h"

template<typename dType>
__global__ void _mul2(int n, dType alpha, dType *a, dType beta, dType *b, dType *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dType ai = (alpha == 1 ? a[i] : alpha == -1 ? 1/a[i] : pow(a[i], alpha));
    dType bi = (beta  == 1 ? b[i] : beta  == -1 ? 1/b[i] : pow(b[i], beta));
    z[i] = ai * bi;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void mul2_32(int n, float  alpha, float  *a, float  beta, float  *b,  float *c) KCALL(_mul2,n,alpha,a,beta,b,c);
  void mul2_64(int n, double alpha, double *a, double beta, double *b, double *c) KCALL(_mul2,n,alpha,a,beta,b,c);
}
