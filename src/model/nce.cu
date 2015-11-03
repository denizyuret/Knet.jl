#include "../knet.h"

template<typename dType>
__global__ void _nce_grad_real(int n, dType *ypred, dType *kqvec, dType *ygrad) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while(i < n) {
    int ij = n*i+i;
    ygrad[ij] = -(kqvec[i]/(exp(ypred[ij]) + kqvec[i]))/n;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void nce_grad_real_32(int n, float  *ypred, float  *kqvec, float  *ygrad) KCALL(_nce_grad_real,n,ypred,kqvec,ygrad);
  void nce_grad_real_64(int n, double *ypred, double *kqvec, double *ygrad) KCALL(_nce_grad_real,n,ypred,kqvec,ygrad);
}

template<typename dType>
__global__ void _nce_loss_real(int n, dType *ypred, dType *kqvec, dType *ytemp) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while(i < n) {
    dType s = ypred[n*i+i];
    ytemp[i] = log(exp(s) + kqvec[i]) - s;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void nce_loss_real_32(int n, float  *ypred, float  *kqvec, float  *ytemp) KCALL(_nce_loss_real,n,ypred,kqvec,ytemp);
  void nce_loss_real_64(int n, double *ypred, double *kqvec, double *ytemp) KCALL(_nce_loss_real,n,ypred,kqvec,ytemp);
}

template<typename dType>
__global__ void _nce_loss_noise(int K, int B, dType *ypred, dType *kqvec, dType *ytemp) {
  int kb = threadIdx.x + blockIdx.x * blockDim.x;
  int KB = K*B;
  while(kb < KB) {
    dType s = ypred[kb];
    dType kq = kqvec[kb % K];
    ytemp[kb] = -log(kq)+log(exp(s)+kq);
    kb += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void nce_loss_noise_32(int K, int B, float  *ypred, float  *kqvec, float  *ytemp) KCALL(_nce_loss_noise,K,B,ypred,kqvec,ytemp);
  void nce_loss_noise_64(int K, int B, double *ypred, double *kqvec, double *ytemp) KCALL(_nce_loss_noise,K,B,ypred,kqvec,ytemp);
}

template<typename dType>
__global__ void _nce_grad_noise(int K, int B, dType *ypred, dType *kqvec, dType *ygrad) {
  int kb = threadIdx.x + blockIdx.x * blockDim.x;
  int KB = K*B;
  while(kb < KB) {
    dType exps = exp(ypred[kb]);
    dType kq = kqvec[kb % K];
    ygrad[kb] = (exps/(exps+kq))/B;
    kb += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void nce_grad_noise_32(int K, int B, float  *ypred, float  *kqvec, float  *ygrad) KCALL(_nce_grad_noise,K,B,ypred,kqvec,ygrad);
  void nce_grad_noise_64(int K, int B, double *ypred, double *kqvec, double *ygrad) KCALL(_nce_grad_noise,K,B,ypred,kqvec,ygrad);
}

