#include "../knet.h"

template<typename dType>
__global__ void _bernoulli(int n, dType p, dType s, dType *x) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[i] = (x[i] <= p ? s : 0);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void bernoulli32(int n, float  p, float  s, float  *x) KCALL(_bernoulli,n,p,s,x);
  void bernoulli64(int n, double p, double s, double *x) KCALL(_bernoulli,n,p,s,x);
}
