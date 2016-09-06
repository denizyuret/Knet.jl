#include "knet.h"

template<typename dType>
__global__ void _axpbforw(int n, dType a, dType *x, dType p, dType b, dType *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dType yi = x[i];
    if (p != 1) yi = pow(yi,p);
    if (a != 1) yi *= a;
    if (b != 0) yi += b;
    y[i] = yi;
    i += blockDim.x * gridDim.x;
  }
}

template<typename dType>
__global__ void _axpbback(int n, dType a, dType *x, dType p, dType *dy, dType *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  dType ap = a*p;   
  while (i < n) {
    dType dxi = dy[i];
    if (a!=1 || p!=1) {
      if (ap != 1) dxi *= ap;
      if (p != 1) dxi *= pow(x[i],p-1);
    }
    dx[i] = dxi;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void axpbforw32(int n, float a, float *x, float p, float b, float *y) KCALL(_axpbforw,n,a,x,p,b,y);
  void axpbforw64(int n, double a, double *x, double p, double b, double *y) KCALL(_axpbforw,n,a,x,p,b,y);
  void axpbback32(int n, float  a, float  *x, float  p, float  *dy, float  *dx) KCALL(_axpbback,n,a,x,p,dy,dx);
  void axpbback64(int n, double a, double *x, double p, double *dy, double *dx) KCALL(_axpbback,n,a,x,p,dy,dx);
}
