#include "../knet.h"

template<typename dType>
__global__ void _mul(int n, dType alpha, dType *a, dType beta, dType *b, dType *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dType ai = (alpha == 1 ? a[i] : alpha == -1 ? 1/a[i] : pow(a[i], alpha));
    dType bi = (beta  == 1 ? b[i] : beta  == -1 ? 1/b[i] : pow(b[i], beta));
    z[i] = ai * bi;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void mul32(int n, float  alpha, float  *a, float  beta, float  *b,  float *c) KCALL(_mul,n,alpha,a,beta,b,c);
  void mul64(int n, double alpha, double *a, double beta, double *b, double *c) KCALL(_mul,n,alpha,a,beta,b,c);
}

// broadcasting mul: c = a^alpha * b^beta
// each dim of a is either 1 or matches b.  c has the same size as b.
template<typename dType>
__global__ void _bmul(int ndims, dType alpha, int *adims, dType *a, dType beta, int *bdims, dType *b, dType *c) {
  int b0, b1, b2, b3, b4, b5, b6, b7, i, j, ai;
  int bi = threadIdx.x + blockIdx.x * blockDim.x;
  int bn = 1;
  for (int n=0; n<ndims; n++) bn *= bdims[n];
  while(bi < bn) {
    j = bi;
    if (ndims > 0) { i=j; j=i/bdims[0]; b0=i-j*bdims[0]; }
    if (ndims > 1) { i=j; j=i/bdims[1]; b1=i-j*bdims[1]; }
    if (ndims > 2) { i=j; j=i/bdims[2]; b2=i-j*bdims[2]; }
    if (ndims > 3) { i=j; j=i/bdims[3]; b3=i-j*bdims[3]; }
    if (ndims > 4) { i=j; j=i/bdims[4]; b4=i-j*bdims[4]; }
    if (ndims > 5) { i=j; j=i/bdims[5]; b5=i-j*bdims[5]; }
    if (ndims > 6) { i=j; j=i/bdims[6]; b6=i-j*bdims[6]; }
    if (ndims > 7) { i=j; j=i/bdims[7]; b7=i-j*bdims[7]; }
    ai = 0;
    if (ndims > 7) { ai = adims[7]*ai + (adims[7]==1 ? 0 : b7); }
    if (ndims > 6) { ai = adims[6]*ai + (adims[6]==1 ? 0 : b6); }
    if (ndims > 5) { ai = adims[5]*ai + (adims[5]==1 ? 0 : b5); }
    if (ndims > 4) { ai = adims[4]*ai + (adims[4]==1 ? 0 : b4); }
    if (ndims > 3) { ai = adims[3]*ai + (adims[3]==1 ? 0 : b3); }
    if (ndims > 2) { ai = adims[2]*ai + (adims[2]==1 ? 0 : b2); }
    if (ndims > 1) { ai = adims[1]*ai + (adims[1]==1 ? 0 : b1); }
    if (ndims > 0) { ai = adims[0]*ai + (adims[0]==1 ? 0 : b0); }
    dType aval = (alpha == 1 ? a[ai] : alpha == -1 ? 1/a[ai] : pow(a[ai], alpha)); // Note the extra work here, a tmp array for a^alpha would reduce the pow ops
    dType bval = (beta  == 1 ? b[bi] : beta  == -1 ? 1/b[bi] : pow(b[bi], beta));
    c[bi] = aval * bval;
    bi += blockDim.x * gridDim.x;
  }
}

// slightly more optimized 2D version
template<typename dType>
__global__ void _bmul2d(dType alpha, int *adims, dType *a, dType beta, int *bdims, dType *b, dType *c) {
  int b0, b1, i, j, ai, A0, A1, B0, B1;
  B0 = bdims[0]; B1 = bdims[1]; A0 = adims[0]; A1 = adims[1];
  int bi = threadIdx.x + blockIdx.x * blockDim.x;
  int bn = B0*B1;
  while(bi < bn) {
    j=bi/B0; b0=bi-j*B0;
    i=j; j=i/B1; b1=i-j*B1;
    ai = A0*(A1==1 ? 0 : b1) + (A0==1 ? 0 : b0);
    dType aval = (alpha == 1 ? a[ai] : alpha == -1 ? 1/a[ai] : pow(a[ai], alpha)); // Note the extra work here, a tmp array for a^alpha would reduce the pow ops
    dType bval = (beta  == 1 ? b[bi] : beta  == -1 ? 1/b[bi] : pow(b[bi], beta));
    c[bi] = aval * bval;
    bi += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void bmul32(int ndims, float  alpha, int *adims, float  *a, float  beta, int *bdims, float  *b, float  *c) {
    if (ndims==2) {
      KCALL(_bmul2d,alpha,adims,a,beta,bdims,b,c);
    } else {
      KCALL(_bmul,ndims,alpha,adims,a,beta,bdims,b,c);
    }
  }
  void bmul64(int ndims, double alpha, int *adims, double *a, double beta, int *bdims, double *b, double *c) {
    if (ndims==2) {
      KCALL(_bmul2d,alpha,adims,a,beta,bdims,b,c);
    } else {
      KCALL(_bmul,ndims,alpha,adims,a,beta,bdims,b,c);
    }
  }
}
