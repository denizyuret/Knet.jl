#include "../knet.h"

template<typename dType>
__global__ void _nceforw(int m, int n, dType *kq, dType *s, dType *p) {
  int ij = threadIdx.x + blockIdx.x * blockDim.x;
  int mn = m*n;
  while(ij < mn) {
    int i = ij % m;
    dType exps = exp(s[ij]);
    p[ij] = exps/(exps+kq[i]);
    ij += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void nceforw32(int m, int n, float  *kq, float  *s, float  *p) KCALL(_nceforw,m,n,kq,s,p);
  void nceforw64(int m, int n, double *kq, double *s, double *p) KCALL(_nceforw,m,n,kq,s,p);
}
