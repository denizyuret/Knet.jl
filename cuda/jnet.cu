// nvcc --shared --compiler-options -fPIC -o libjnet.so jnet.cu -lcublas
// TODO:
// train
// update: check L2, momentum, nesterov
// update: implement maxnorm, L1
// dropout
// compare with caffe, matlab

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <assert.h>
#include "jnet.h"
#define BLK 128
#define THR 128

static inline float *gpuArray(int nfloats);
static inline float *gpuCopy(int nfloats, float *cpuArray);
static inline float *gpuFill(int nfloats, float val);
static inline void fforw(Layer l);
static inline float *initforw(Layer l, float *x, int xcols);
static inline void fback(Layer l);
static inline float *initback(Layer l, float *dy, int dx);
static inline void initupdate(Layer l);
__global__ void _reluforw(int n, float *y);
__global__ void _reluback(int n, float *y, float *dy);
__global__ void _softback(int nrows, int ncols, float *y, float *dy);
__global__ void _l1reg(int n, float l1, float *w, float *dw);
__global__ void _adagrad(int n, float *dw2, float *dw);
__global__ void _fill(int n, float val, float *x);

#define CUDA(_s) assert((_s) == cudaSuccess)
#define CUBLAS(_s) assert(((_s), cublasGetError())==CUBLAS_STATUS_SUCCESS)

#define gpuFree(x) CUDA(cudaFree(x))
#define gpuGetMatrix(rows,cols,from,to) CUBLAS(cublasGetMatrix(rows,cols,sizeof(float),from,rows,to,rows))
#define gpuSetMatrix(rows,cols,from,to) CUBLAS(cublasSetMatrix(rows,cols,sizeof(float),from,rows,to,rows))

static inline float *gpuArray(int nfloats) {
  float *gptr;
  CUDA(cudaMalloc((void **) &gptr, nfloats * sizeof(float)));
  return gptr;
}

static inline float *gpuCopy(int nfloats, float *cpuArray) {
  float *gptr = gpuArray(nfloats);
  CUDA(cudaMemcpy(gptr, cpuArray, nfloats * sizeof(float), cudaMemcpyHostToDevice));
  return gptr;
}

static inline float *gpuFill(int nfloats, float val) {
  float *gptr = gpuArray(nfloats);
  _fill<<<BLK,THR>>>(nfloats, val, gptr);
  return gptr;
}


Layer layer(LayerType type, int wrows, int wcols, float *w, float *b) {
  static int init = 0;
  if (!init) { CUBLAS(cublasInit()); init=1; }
  Layer l = (Layer) calloc(1, sizeof(struct LayerS));
  assert(l != NULL);
  l->type = type;
  l->wrows = wrows;
  l->wcols = wcols;
  assert(w != NULL);
  l->w = gpuCopy(wrows*wcols, w);
  if (b != NULL) l->b = gpuCopy(wrows, b);
  return(l);
}

void lfree(Layer l) {
  gpuFree(l->w);
  gpuFree(l->b);
  //gpuFree(l->x);   // taken as input, not alloced
  gpuFree(l->y);   
  gpuFree(l->xones);
  //gpuFree(l->xmask);

  gpuFree(l->dw);
  gpuFree(l->db);
  gpuFree(l->dx);
  //gpuFree(l->dy);  // taken as input, not alloced

  gpuFree(l->dw1);
  gpuFree(l->dw2);
  gpuFree(l->db1);
  gpuFree(l->db2);
  free(l);
}

void set_adagrad(Layer l, int i) { l->adagrad = i; }
void set_nesterov(Layer l, int i) { l->nesterov = i; }
void set_learningRate(Layer l, float f) { l->learningRate = f; }
void set_momentum(Layer l, float f) { l->momentum = f; }
void set_dropout(Layer l, float f) { l->dropout = f; }
void set_maxnorm(Layer l, float f) { l->maxnorm = f; }
void set_L1(Layer l, float f) { l->L1 = f; }
void set_L2(Layer l, float f) { l->L2 = f; }

int lsize(Layer l, int i) { 
  return (i==1 ? l->wrows : i==2 ? l->wcols : 1);
}

Layer relu(int wrows, int wcols, float *w, float *b) {
  return layer(RELU, wrows, wcols, w, b);
}

Layer soft(int wrows, int wcols, float *w, float *b) {
  return layer(SOFT, wrows, wcols, w, b);
}

void forward(Layer *net, float *x, float *y, int nlayer, int xcols, int batch) {
  int xrows = net[0]->wcols;
  int yrows = net[nlayer-1]->wrows;
  float *xgpu = gpuArray(xrows * batch);
  for (int b = 0; b < xcols; b += batch) {
    if (b + batch > xcols) batch = xcols - b;
    gpuSetMatrix(xrows, batch, &x[b*xrows], xgpu);
    float *gptr = xgpu;
    for (int l = 0; l < nlayer; l++)
      gptr = lforw(net[l], gptr, batch);
    gpuGetMatrix(yrows, batch, gptr, &y[b*yrows]);
  }
  gpuFree(xgpu);
}

void forwback(Layer *net, float *x, float *y, int nlayer, int xcols, int batch) {
  int xrows = net[0]->wcols;
  int yrows = net[nlayer-1]->wrows;
  float *xgpu = gpuArray(xrows * batch);
  float *ygpu = gpuArray(yrows * batch);
  for (int b = 0; b < xcols; b += batch) {
    if (b + batch > xcols) batch = xcols - b;
    gpuSetMatrix(xrows, batch, &x[b*xrows], xgpu);
    float *gptr = xgpu;
    for (int l = 0; l < nlayer; l++)
      gptr = lforw(net[l], gptr, batch);
    gpuSetMatrix(yrows, batch, &y[b*yrows], ygpu);
    gptr = ygpu;
    for (int l = nlayer - 1; l >= 0; l--)
      gptr = lback(net[l], gptr, (l>0));
  }
  gpuFree(xgpu);
  gpuFree(ygpu);
}


float *lforw(Layer l, float *x, int xcols) {
  // We assume x is already a device pointer.
  // Otherwise we'd have to do unnecessary copying between layers.
  // We assume xrows == l->wcols and x is column-major.  
  l->x = initforw(l, x, xcols);

  // y = w * x
  // gemm(opA,opB,m,n,k,α,A(m,k),lda=m,B(k,n),ldb=k,β,C(m,n),ldc=m): C = α op(A) op(B) + β C
  CUBLAS(cublasSgemm('N', 'N', l->wrows, l->xcols, l->wcols, 1.0, l->w, l->wrows, l->x, l->wcols, 0.0, l->y, l->wrows));

  if (l->b != NULL) {
    // y = y + b  with singleton expansion
    // ger(m,n,α,x(m),incx=1,y(n),incy=1,A(m,n),lda=m): A = α x y' + A
    CUBLAS(cublasSger(l->wrows, l->xcols, 1.0, l->b, 1, l->xones, 1, l->y, l->wrows));
  }
  // y = f(y) where f is relu, sigm etc.
  fforw(l);
  return l->y;
}

static inline void fforw(Layer l) {
  switch(l->type) {
  case RELU:
    _reluforw<<<BLK,THR>>>(l->wrows * l->xcols, l->y);
    CUDA(cudaGetLastError());
    break;
  }
}

static inline float *initforw(Layer l, float *x, int xcols) {
  // Alloc/realloc l->y and l->xones if necessary and update l->xcols
  int yrows = l->wrows;
  int ycols = xcols;
  if ((l->y == NULL) || (l->xcols != xcols)) {
    gpuFree(l->y);
    l->y = gpuArray(yrows * ycols);
  }
  if ((l->b != NULL) && ((l->xones == NULL) || l->xcols != xcols)) {
    gpuFree(l->xones);
    l->xones = gpuFill(xcols, 1.0);
  }
  if ((l->dx != NULL) && (l->xcols != xcols)) {
    gpuFree(l->dx);
    l->dx = NULL;	/* to be reallocated */
  }
  l->xcols = xcols;
  return x;
}

float *lback(Layer l, float *dy, int dx) {
  // We assume dy is already a device pointer.
  // Otherwise we'd have to do unnecessary copying between layers.
  // We assume dy has the same size as l.y: (wrows,xcols)
  l->dy = initback(l, dy, dx);
  
  // dy = fback(dy) where fback is the derivative of fforw
  fback(l);

  // dw = dy * x'
  // gemm(opA,opB,m,n,k,α,A(m,k),lda=m,B(k,n),ldb=k,β,C(m,n),ldc=m): C = α op(A) op(B) + β C
  // m = wrows; n = wcols; k = xcols
  CUBLAS(cublasSgemm('N', 'T', l->wrows, l->wcols, l->xcols, 1.0, l->dy, l->wrows, l->x, l->wcols, 0.0, l->dw, l->wrows));

  if (l->b != NULL) {
    // db = sum(dy,2) = dy * ones
    // gemv(op,m,n,α,A(m,n),lda=m,x(n),incx=1,β,y(m),incy=1): y = α op(A) x + β y
    CUBLAS(cublasSgemv('N', l->wrows, l->xcols, 1.0, l->dy, l->wrows, l->xones, 1, 0.0, l->db, 1));
  }
  if (dx) { // dx is optional because it is expensive and unnecessary for input layer
    // dx=w' * dy
    // gemm(opA,opB,m,n,k,α,A(m,k),lda=m,B(k,n),ldb=k,β,C(m,n),ldc=m): C = α op(A) op(B) + β C
    // m = wcols, n = xcols, k = wrows
    CUBLAS(cublasSgemm('T', 'N', l->wcols, l->xcols, l->wrows, 1.0, l->w, l->wrows, l->dy, l->wrows, 0.0, l->dx, l->wcols));
  }
  return l->dx;
}

static inline void fback(Layer l) {
  switch(l->type) {
  case RELU:
    _reluback<<<BLK,THR>>>(l->wrows * l->xcols, l->y, l->dy);
    CUDA(cudaGetLastError());
    break;
  case SOFT:
    _softback<<<BLK,THR>>>(l->wrows, l->xcols, l->y, l->dy);
    CUDA(cudaGetLastError());
    break;
  }
}

static inline float *initback(Layer l, float *dy, int dx) {
  if (l->dw == NULL) l->dw = gpuArray(l->wrows * l->wcols);
  if ((l->b != NULL) && (l->db == NULL)) l->db = gpuArray(l->wrows);
  if (dx && (l->dx == NULL)) l->dx = gpuArray(l->wcols * l->xcols);
  return dy;
}

void update(Layer l) {
  initupdate(l);
  if (l->learningRate == 0) return;
  int nw = l->wcols * l->wrows;
  int nb = (l->b == NULL ? 0 : l->wrows);

  if (l->L1 != 0) {
    /* L1 regularization:
       J(w,b) = Jerr + L1 Σ|wi|
       ∂J/∂wi = ∂Jerr/∂wi + L1 sign(wi)
       dw contains ∂Jerr/∂wi after lback
       we want: dw += L1 * sign(w)
       axpy(n,α,x(n),incx=1,y(n),incy=1): y = α x + y
    */
    _l1reg<<<BLK,THR>>>(nw, l->L1, l->w, l->dw);
    CUDA(cudaGetLastError());
  }

  if (l->L2 != 0) { 
    /* L2 regularization:
       J(w,b) = Jerr + (L2/2)|w|^2
       ∂J/∂wi = ∂Jerr/∂wi + L2 wi 
       dw contains ∂Jerr/∂wi after lback
       we want: dw += L2 * w
       axpy(n,α,x(n),incx=1,y(n),incy=1): y = α x + y
    */
    cublasSaxpy(nw, l->L2, l->w, 1, l->dw, 1);
  }
  if (l->adagrad) {
    /* ADAGRAD:
       dw2 += dw.*dw 
       dw /= (epsilon + sqrt(dw2))
       and similarly for db.
    */
    _adagrad<<<BLK,THR>>>(nw, l->dw2, l->dw);
    if (nb) { _adagrad<<<BLK,THR>>>(nb, l->db2, l->db); }
    CUDA(cudaGetLastError());
  }
  if (l->learningRate != 1) {
    /* LearningRate:
       Scale dw and db with the learning rate.
       dw,db *= learningRate
       scal(n,α,x(n),incx=1): x = α x
    */
    cublasSscal(nw, l->learningRate, l->dw, 1);
    if (nb) cublasSscal(nb, l->learningRate, l->db, 1);
  }
  if (l->momentum != 0) {  
    /* Momentum:
       why do we apply it here?
       do we apply it to db?
       check the following:
       dw1 = momentum * dw1 + dw
       dw = dw1   :without nesterov
       dw = momentum * dw1 + dw   :with nesterov
    */
    assert(1==0);	/* need to check first */
    cublasSscal(nw, l->momentum, l->dw1, 1);
    cublasSaxpy(nw, 1.0, l->dw, 1, l->dw1, 1);
    if (l->nesterov) {
      cublasSaxpy(nw, l->momentum, l->dw1, 1, l->dw, 1);
    } else {
      cublasScopy(nw, l->dw1, 1, l->dw, 1);
    }
  }
  /* Finally apply gradient descent: w -= dw, b -= db */
  cublasSaxpy(nw, -1.0, l->dw, 1, l->w, 1);
  if (nb) cublasSaxpy(nb, -1.0, l->db, 1, l->b, 1);

  if (l->maxnorm != 0) {
    /* MaxNorm:

     */
    assert(1 == 0);  // TBD
  }
}

static inline void initupdate(Layer l) {
  if (l->adagrad) {
    if (l->dw2 == NULL) l->dw2 = gpuFill(l->wrows * l->wcols, 0.0);
    if ((l->b != NULL) && (l->db2 == NULL)) l->db2 = gpuFill(l->wrows, 0.0);
  }
  if (l->momentum != 0) {
    if (l->dw1 == NULL) l->dw1 = gpuFill(l->wrows * l->wcols, 0.0);
    if ((l->b != NULL) && (l->db1 == NULL)) l->db1 = gpuFill(l->wrows, 0.0);
  }
}


__global__ void _fill(int n, float val, float *x) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[i] = val;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _adagrad(int n, float *dw2, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dw2[i] += dw[i] * dw[i];
    dw[i] /= (1e-8 + sqrt(dw2[i]));
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _l1reg(int n, float l1, float *w, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dw[i] += (w[i] >= 0 ? l1 : -l1);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _reluforw(int n, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] < 0) y[i] = 0;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _reluback(int n, float *y, float *dy) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] <= 0) dy[i] = 0;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _softback(int nrows, int ncols, float *y, float *dy) {
  float y0, sum;
  int i0, i1;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  while (col < ncols) {
    i0 = col * nrows;
    i1 = i0  + nrows;
    y0 = -INFINITY;
    //y0 = y[i0];
    for (int i=i0; i<i1; i++) {
      if (y[i] > y0) {
	y0 = y[i];
      }
    }
    sum = 0;
    for (int i=i0; i<i1; i++) {
      y[i] = exp(y[i]-y0);
      sum += y[i];
    }
    for (int i=i0; i<i1; i++) {
      y[i] /= sum;
      dy[i] = (y[i] - dy[i]) / ncols;
    }
    col += blockDim.x * gridDim.x;
  }
}

