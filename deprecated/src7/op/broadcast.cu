#include "../knet.h"
#include <assert.h>

// broadcasting add: c = a .+ b
// adims, bdims, cdims each have ndims elements.
// each dim of a and b is either 1 or matches c.
// c has celts elements, i.e. prod(cdims)=celts
// arrays assumed to be column major as in Julia (fastest dims first)
template<typename dType>
__global__ void _badd3forw(int celts, int ndims, int *adims, dType *a, int *bdims, dType *b, int *cdims, dType *c) {
  int c0, c1, c2, c3, c4, c5, c6, c7, i, j, ai, bi; // using locals instead of array for register speed
  int ci = threadIdx.x + blockIdx.x * blockDim.x;
  while(ci < celts) {
    ai = bi = 0;
    j = ci;
    if (ndims > 0) { i=j; j=i/cdims[0]; c0=i-j*cdims[0]; 
      if (ndims > 1) { i=j; j=i/cdims[1]; c1=i-j*cdims[1]; 
	if (ndims > 2) { i=j; j=i/cdims[2]; c2=i-j*cdims[2]; 
	  if (ndims > 3) { i=j; j=i/cdims[3]; c3=i-j*cdims[3]; 
	    if (ndims > 4) { i=j; j=i/cdims[4]; c4=i-j*cdims[4]; 
	      if (ndims > 5) { i=j; j=i/cdims[5]; c5=i-j*cdims[5]; 
		if (ndims > 6) { i=j; j=i/cdims[6]; c6=i-j*cdims[6]; 
		  if (ndims > 7) { i=j; j=i/cdims[7]; c7=i-j*cdims[7]; 
		    ai = adims[7]*ai + (adims[7]==1 ? 0 : c7); bi = bdims[7]*bi + (bdims[7]==1 ? 0 : c7); }
		  ai = adims[6]*ai + (adims[6]==1 ? 0 : c6); bi = bdims[6]*bi + (bdims[6]==1 ? 0 : c6); }
		ai = adims[5]*ai + (adims[5]==1 ? 0 : c5); bi = bdims[5]*bi + (bdims[5]==1 ? 0 : c5); }
	      ai = adims[4]*ai + (adims[4]==1 ? 0 : c4); bi = bdims[4]*bi + (bdims[4]==1 ? 0 : c4); }
	    ai = adims[3]*ai + (adims[3]==1 ? 0 : c3); bi = bdims[3]*bi + (bdims[3]==1 ? 0 : c3); }
	  ai = adims[2]*ai + (adims[2]==1 ? 0 : c2); bi = bdims[2]*bi + (bdims[2]==1 ? 0 : c2); }
	ai = adims[1]*ai + (adims[1]==1 ? 0 : c1); bi = bdims[1]*bi + (bdims[1]==1 ? 0 : c1); }
      ai = adims[0]*ai + (adims[0]==1 ? 0 : c0); bi = bdims[0]*bi + (bdims[0]==1 ? 0 : c0); }
    c[ci] = a[ai] + b[bi];
    ci += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void badd3forw32(int celts, int ndims, int *adims, float  *a, int *bdims, float  *b, int *cdims, float  *c) KCALL(_badd3forw,celts,ndims,adims,a,bdims,b,cdims,c);
  void badd3forw64(int celts, int ndims, int *adims, double *a, int *bdims, double *b, int *cdims, double *c) KCALL(_badd3forw,celts,ndims,adims,a,bdims,b,cdims,c);
}

// broadcasting multiplication
template<typename dType>
__global__ void _bmul3forw(int celts, int ndims, int *adims, dType *a, int *bdims, dType *b, int *cdims, dType *c) {
  int c0, c1, c2, c3, c4, c5, c6, c7, i, j, ai, bi; // using locals instead of array for register speed
  int ci = threadIdx.x + blockIdx.x * blockDim.x;
  while(ci < celts) {
    ai = bi = 0;
    j = ci;
    if (ndims > 0) { i=j; j=i/cdims[0]; c0=i-j*cdims[0]; 
      if (ndims > 1) { i=j; j=i/cdims[1]; c1=i-j*cdims[1]; 
	if (ndims > 2) { i=j; j=i/cdims[2]; c2=i-j*cdims[2]; 
	  if (ndims > 3) { i=j; j=i/cdims[3]; c3=i-j*cdims[3]; 
	    if (ndims > 4) { i=j; j=i/cdims[4]; c4=i-j*cdims[4]; 
	      if (ndims > 5) { i=j; j=i/cdims[5]; c5=i-j*cdims[5]; 
		if (ndims > 6) { i=j; j=i/cdims[6]; c6=i-j*cdims[6]; 
		  if (ndims > 7) { i=j; j=i/cdims[7]; c7=i-j*cdims[7]; 
		    ai = adims[7]*ai + (adims[7]==1 ? 0 : c7); bi = bdims[7]*bi + (bdims[7]==1 ? 0 : c7); }
		  ai = adims[6]*ai + (adims[6]==1 ? 0 : c6); bi = bdims[6]*bi + (bdims[6]==1 ? 0 : c6); }
		ai = adims[5]*ai + (adims[5]==1 ? 0 : c5); bi = bdims[5]*bi + (bdims[5]==1 ? 0 : c5); }
	      ai = adims[4]*ai + (adims[4]==1 ? 0 : c4); bi = bdims[4]*bi + (bdims[4]==1 ? 0 : c4); }
	    ai = adims[3]*ai + (adims[3]==1 ? 0 : c3); bi = bdims[3]*bi + (bdims[3]==1 ? 0 : c3); }
	  ai = adims[2]*ai + (adims[2]==1 ? 0 : c2); bi = bdims[2]*bi + (bdims[2]==1 ? 0 : c2); }
	ai = adims[1]*ai + (adims[1]==1 ? 0 : c1); bi = bdims[1]*bi + (bdims[1]==1 ? 0 : c1); }
      ai = adims[0]*ai + (adims[0]==1 ? 0 : c0); bi = bdims[0]*bi + (bdims[0]==1 ? 0 : c0); }
    c[ci] = a[ai] * b[bi];
    ci += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void bmul3forw32(int celts, int ndims, int *adims, float  *a, int *bdims, float  *b, int *cdims, float  *c) KCALL(_bmul3forw,celts,ndims,adims,a,bdims,b,cdims,c);
  void bmul3forw64(int celts, int ndims, int *adims, double *a, int *bdims, double *b, int *cdims, double *c) KCALL(_bmul3forw,celts,ndims,adims,a,bdims,b,cdims,c);
}

// broadcast add with a single array (the other assumed zero)
template<typename dType>
__global__ void _badd2forw(int celts, int ndims, int *adims, dType *a, int *cdims, dType *c) {
  int c0, c1, c2, c3, c4, c5, c6, c7, i, j, ai;
  int ci = threadIdx.x + blockIdx.x * blockDim.x;
  while(ci < celts) {
    j = ci; ai = 0;
    if (ndims > 0) { i=j; j=i/cdims[0]; c0=i-j*cdims[0]; 
      if (ndims > 1) { i=j; j=i/cdims[1]; c1=i-j*cdims[1]; 
	if (ndims > 2) { i=j; j=i/cdims[2]; c2=i-j*cdims[2]; 
	  if (ndims > 3) { i=j; j=i/cdims[3]; c3=i-j*cdims[3]; 
	    if (ndims > 4) { i=j; j=i/cdims[4]; c4=i-j*cdims[4]; 
	      if (ndims > 5) { i=j; j=i/cdims[5]; c5=i-j*cdims[5]; 
		if (ndims > 6) { i=j; j=i/cdims[6]; c6=i-j*cdims[6]; 
		  if (ndims > 7) { i=j; j=i/cdims[7]; c7=i-j*cdims[7]; 
		    ai = adims[7]*ai + (adims[7]==1 ? 0 : c7); }
		  ai = adims[6]*ai + (adims[6]==1 ? 0 : c6); }
		ai = adims[5]*ai + (adims[5]==1 ? 0 : c5); }
	      ai = adims[4]*ai + (adims[4]==1 ? 0 : c4); }
	    ai = adims[3]*ai + (adims[3]==1 ? 0 : c3); }
	  ai = adims[2]*ai + (adims[2]==1 ? 0 : c2); }
	ai = adims[1]*ai + (adims[1]==1 ? 0 : c1); }
      ai = adims[0]*ai + (adims[0]==1 ? 0 : c0); }
    c[ci] = a[ai];
    ci += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void badd2forw32(int celts, int ndims, int *adims, float  *a, int *cdims, float  *c) KCALL(_badd2forw,celts,ndims,adims,a,cdims,c);
  void badd2forw64(int celts, int ndims, int *adims, double *a, int *cdims, double *c) KCALL(_badd2forw,celts,ndims,adims,a,cdims,c);
}

// broadcast add back pass, no need for two arg version, single enough
template<typename dType>
__global__ void _badd2back(int celts, int ndims, int *cdims, dType *dc, int *adims, dType *da) {
  int c0, c1, c2, c3, c4, c5, c6, c7, i, j, ai;
  int ci = threadIdx.x + blockIdx.x * blockDim.x;
  while(ci < celts) {
    j = ci; ai = 0;
    if (ndims > 0) { i=j; j=i/cdims[0]; c0=i-j*cdims[0]; 
      if (ndims > 1) { i=j; j=i/cdims[1]; c1=i-j*cdims[1]; 
	if (ndims > 2) { i=j; j=i/cdims[2]; c2=i-j*cdims[2]; 
	  if (ndims > 3) { i=j; j=i/cdims[3]; c3=i-j*cdims[3]; 
	    if (ndims > 4) { i=j; j=i/cdims[4]; c4=i-j*cdims[4]; 
	      if (ndims > 5) { i=j; j=i/cdims[5]; c5=i-j*cdims[5]; 
		if (ndims > 6) { i=j; j=i/cdims[6]; c6=i-j*cdims[6]; 
		  if (ndims > 7) { i=j; j=i/cdims[7]; c7=i-j*cdims[7]; 
		    ai = adims[7]*ai + (adims[7]==1 ? 0 : c7); }
		  ai = adims[6]*ai + (adims[6]==1 ? 0 : c6); }
		ai = adims[5]*ai + (adims[5]==1 ? 0 : c5); }
	      ai = adims[4]*ai + (adims[4]==1 ? 0 : c4); }
	    ai = adims[3]*ai + (adims[3]==1 ? 0 : c3); }
	  ai = adims[2]*ai + (adims[2]==1 ? 0 : c2); }
	ai = adims[1]*ai + (adims[1]==1 ? 0 : c1); }
      ai = adims[0]*ai + (adims[0]==1 ? 0 : c0); }
    atomicAdd(&da[ai], dc[ci]);
    ci += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void badd2back32(int celts, int ndims, int *cdims, float  *dc, int *adims, float  *da) KCALL(_badd2back,celts,ndims,cdims,dc,adims,da);
  void badd2back64(int celts, int ndims, int *cdims, double *dc, int *adims, double *da) KCALL(_badd2back,celts,ndims,cdims,dc,adims,da);
}

// broadcasting mul back pass
template<typename dType>
__global__ void _bmul3back(int celts, int ndims, int *cdims, dType *dc, int *adims, dType *a, int *bdims, dType *db) {
  int c0, c1, c2, c3, c4, c5, c6, c7, i, j, ai, bi; // using locals instead of array for register speed
  int ci = threadIdx.x + blockIdx.x * blockDim.x;
  while(ci < celts) {
    ai = bi = 0;
    j = ci;
    if (ndims > 0) { i=j; j=i/cdims[0]; c0=i-j*cdims[0]; 
      if (ndims > 1) { i=j; j=i/cdims[1]; c1=i-j*cdims[1]; 
	if (ndims > 2) { i=j; j=i/cdims[2]; c2=i-j*cdims[2]; 
	  if (ndims > 3) { i=j; j=i/cdims[3]; c3=i-j*cdims[3]; 
	    if (ndims > 4) { i=j; j=i/cdims[4]; c4=i-j*cdims[4]; 
	      if (ndims > 5) { i=j; j=i/cdims[5]; c5=i-j*cdims[5]; 
		if (ndims > 6) { i=j; j=i/cdims[6]; c6=i-j*cdims[6]; 
		  if (ndims > 7) { i=j; j=i/cdims[7]; c7=i-j*cdims[7]; 
		    ai = adims[7]*ai + (adims[7]==1 ? 0 : c7); bi = bdims[7]*bi + (bdims[7]==1 ? 0 : c7); }
		  ai = adims[6]*ai + (adims[6]==1 ? 0 : c6); bi = bdims[6]*bi + (bdims[6]==1 ? 0 : c6); }
		ai = adims[5]*ai + (adims[5]==1 ? 0 : c5); bi = bdims[5]*bi + (bdims[5]==1 ? 0 : c5); }
	      ai = adims[4]*ai + (adims[4]==1 ? 0 : c4); bi = bdims[4]*bi + (bdims[4]==1 ? 0 : c4); }
	    ai = adims[3]*ai + (adims[3]==1 ? 0 : c3); bi = bdims[3]*bi + (bdims[3]==1 ? 0 : c3); }
	  ai = adims[2]*ai + (adims[2]==1 ? 0 : c2); bi = bdims[2]*bi + (bdims[2]==1 ? 0 : c2); }
	ai = adims[1]*ai + (adims[1]==1 ? 0 : c1); bi = bdims[1]*bi + (bdims[1]==1 ? 0 : c1); }
      ai = adims[0]*ai + (adims[0]==1 ? 0 : c0); bi = bdims[0]*bi + (bdims[0]==1 ? 0 : c0); }
    atomicAdd(&db[bi], dc[ci] * a[ai]);
    ci += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void bmul3back32(int celts, int ndims, int *cdims, float  *dc, int *adims, float  *a, int *bdims, float  *db) KCALL(_bmul3back,celts,ndims,cdims,dc,adims,a,bdims,db);
  void bmul3back64(int celts, int ndims, int *cdims, double *dc, int *adims, double *a, int *bdims, double *db) KCALL(_bmul3back,celts,ndims,cdims,dc,adims,a,bdims,db);
}

template<typename dType>
__global__ void _mul(int n, dType *a, dType *b, dType *c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    c[i] = a[i] * b[i];
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void mul32(int n, float  *a, float  *b,  float *c) KCALL(_mul,n,a,b,c);
  void mul64(int n, double *a, double *b, double *c) KCALL(_mul,n,a,b,c);
}


// DEAD CODE:

// // slightly more optimized 2D version
// template<typename dType>
// __global__ void _addforw2d(dType alpha, int *adims, dType *a, dType beta, int *bdims, dType *b, dType *c) {
//   int b0, b1, i, j, ai, A0, A1, B0, B1;
//   B0 = bdims[0]; B1 = bdims[1]; A0 = adims[0]; A1 = adims[1];
//   int bi = threadIdx.x + blockIdx.x * blockDim.x;
//   int bn = bdims[0]*bdims[1];
//   while(bi < bn) {
//     j=bi/B0; b0=bi-j*B0;
//     i=j; j=i/B1; b1=i-j*B1;
//     ai = A0*(A1==1 ? 0 : b1) + (A0==1 ? 0 : b0);
//     c[bi] = alpha * a[ai] + beta * b[bi];
//     /* 
//     dType cval = b[bi];
//     if (beta != 1) cval *= beta;
//     if (alpha != 0) {
//       if (alpha != 1) cval += alpha * a[ai];
//       else cval += a[ai];
//     }
//     c[bi] = cval;
//     */
//     bi += blockDim.x * gridDim.x;
//   }
// }

//   void addforw32(int ndims, float  alpha, int *adims, float  *a, float  beta, int *bdims, float  *b, float  *c) {
//     if (ndims==2) {
//       KCALL(_addforw2d,alpha,adims,a,beta,bdims,b,c);
//     } else {
//       KCALL(_addforw,ndims,alpha,adims,a,beta,bdims,b,c);
//     }
//   }
//   void addforw64(int ndims, double alpha, int *adims, double *a, double beta, int *bdims, double *b, double *c) {
//     if (ndims==2) {
//       KCALL(_addforw2d,alpha,adims,a,beta,bdims,b,c);
//     } else {
//       KCALL(_addforw,ndims,alpha,adims,a,beta,bdims,b,c);
//     }
//   }

//   /*
// #include <stdio.h>

//   void addforw32(int ndims, float  alpha, int *adims, float  *a, float  beta, int *bdims, float  *b, float  *c) {
//     printf("ndims=%d alpha=%g beta=%g\n", ndims, alpha, beta);
//     int *ad = (int *) calloc(ndims, sizeof(int));
//     int *bd = (int *) calloc(ndims, sizeof(int));
//     cudaMemcpy(ad, adims, ndims*sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(bd, bdims, ndims*sizeof(int), cudaMemcpyDeviceToHost);
//     for (int n=0; n<ndims; n++) printf("a[%d]=%d b[%d]=%d\n", n, ad[n], n, bd[n]);
//     KCALL(_addforw,ndims,alpha,adims,a,beta,bdims,b,c);
//   }
//   */
// }

