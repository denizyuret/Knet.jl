#include "../knet.h"
#include <assert.h>

// broadcasting add: c = alpha*a + beta * b
// each dim of a is either 1 or matches b.  c has the same size as b.
template<typename dType>
__global__ void _addforw(int ndims, dType alpha, int *adims, dType *a, dType beta, int *bdims, dType *b, dType *c) {
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
    c[bi] = alpha * a[ai] + beta * b[bi];
    bi += blockDim.x * gridDim.x;
  }
}

// slightly more optimized 2D version
template<typename dType>
__global__ void _addforw2d(dType alpha, int *adims, dType *a, dType beta, int *bdims, dType *b, dType *c) {
  int b0, b1, i, j, ai, A0, A1, B0, B1;
  B0 = bdims[0]; B1 = bdims[1]; A0 = adims[0]; A1 = adims[1];
  int bi = threadIdx.x + blockIdx.x * blockDim.x;
  int bn = bdims[0]*bdims[1];
  while(bi < bn) {
    j=bi/B0; b0=bi-j*B0;
    i=j; j=i/B1; b1=i-j*B1;
    ai = A0*(A1==1 ? 0 : b1) + (A0==1 ? 0 : b0);
    c[bi] = alpha * a[ai] + beta * b[bi];
    /* 
    dType cval = b[bi];
    if (beta != 1) cval *= beta;
    if (alpha != 0) {
      if (alpha != 1) cval += alpha * a[ai];
      else cval += a[ai];
    }
    c[bi] = cval;
    */
    bi += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void addforw32(int ndims, float  alpha, int *adims, float  *a, float  beta, int *bdims, float  *b, float  *c) {
    if (ndims==2) {
      KCALL(_addforw2d,alpha,adims,a,beta,bdims,b,c);
    } else {
      KCALL(_addforw,ndims,alpha,adims,a,beta,bdims,b,c);
    }
  }
  void addforw64(int ndims, double alpha, int *adims, double *a, double beta, int *bdims, double *b, double *c) {
    if (ndims==2) {
      KCALL(_addforw2d,alpha,adims,a,beta,bdims,b,c);
    } else {
      KCALL(_addforw,ndims,alpha,adims,a,beta,bdims,b,c);
    }
  }

  /*
#include <stdio.h>

  void addforw32(int ndims, float  alpha, int *adims, float  *a, float  beta, int *bdims, float  *b, float  *c) {
    printf("ndims=%d alpha=%g beta=%g\n", ndims, alpha, beta);
    int *ad = (int *) calloc(ndims, sizeof(int));
    int *bd = (int *) calloc(ndims, sizeof(int));
    cudaMemcpy(ad, adims, ndims*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bd, bdims, ndims*sizeof(int), cudaMemcpyDeviceToHost);
    for (int n=0; n<ndims; n++) printf("a[%d]=%d b[%d]=%d\n", n, ad[n], n, bd[n]);
    KCALL(_addforw,ndims,alpha,adims,a,beta,bdims,b,c);
  }
  */
}

template<typename dType>
__global__ void _addback(int ndims, int *cdims, dType *c, int *adims, dType *a) {
  int c0, c1, c2, c3, c4, c5, c6, c7, i, j, ai;
  int ci = threadIdx.x + blockIdx.x * blockDim.x;
  int cn = 1;
  for (int n=0; n<ndims; n++) cn *= cdims[n];
  while(ci < cn) {
    j = ci;
    if (ndims > 0) { i=j; j=i/cdims[0]; c0=i-j*cdims[0]; }
    if (ndims > 1) { i=j; j=i/cdims[1]; c1=i-j*cdims[1]; }
    if (ndims > 2) { i=j; j=i/cdims[2]; c2=i-j*cdims[2]; }
    if (ndims > 3) { i=j; j=i/cdims[3]; c3=i-j*cdims[3]; }
    if (ndims > 4) { i=j; j=i/cdims[4]; c4=i-j*cdims[4]; }
    if (ndims > 5) { i=j; j=i/cdims[5]; c5=i-j*cdims[5]; }
    if (ndims > 6) { i=j; j=i/cdims[6]; c6=i-j*cdims[6]; }
    if (ndims > 7) { i=j; j=i/cdims[7]; c7=i-j*cdims[7]; }
    ai = 0;
    if (ndims > 7) { ai = adims[7]*ai + (adims[7]==1 ? 0 : c7); }
    if (ndims > 6) { ai = adims[6]*ai + (adims[6]==1 ? 0 : c6); }
    if (ndims > 5) { ai = adims[5]*ai + (adims[5]==1 ? 0 : c5); }
    if (ndims > 4) { ai = adims[4]*ai + (adims[4]==1 ? 0 : c4); }
    if (ndims > 3) { ai = adims[3]*ai + (adims[3]==1 ? 0 : c3); }
    if (ndims > 2) { ai = adims[2]*ai + (adims[2]==1 ? 0 : c2); }
    if (ndims > 1) { ai = adims[1]*ai + (adims[1]==1 ? 0 : c1); }
    if (ndims > 0) { ai = adims[0]*ai + (adims[0]==1 ? 0 : c0); }
    atomicAdd(&a[ai], c[ci]);
    ci += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void addback32(int ndims, int *cdims, float  *c, int *adims, float  *a) KCALL(_addback,ndims,cdims,c,adims,a);
  void addback64(int ndims, int *cdims, double *c, int *adims, double *a) KCALL(_addback,ndims,cdims,c,adims,a);
}
