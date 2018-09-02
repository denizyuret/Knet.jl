// nvcc --shared --compiler-options -fPIC -o libkunet.so kunet.cu -lcublas
// TODO: Make dropout use xmask (renamed to xdrop) instead of modifying incoming x

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <assert.h>
#include "kunet.h"
#define BLK 128
#define THR 128

#define CUDA(_s) assert((_s) == cudaSuccess)
#define CUBLAS(_s) assert((_s) == CUBLAS_STATUS_SUCCESS)
#define CURAND(_s) assert((_s) == CURAND_STATUS_SUCCESS)
#define gpuGetMatrix(rows,cols,from,to) CUDA(cudaMemcpy((to),(from),(rows)*(cols)*sizeof(float),cudaMemcpyDeviceToHost))
#define gpuSetMatrix(rows,cols,from,to) CUDA(cudaMemcpy((to),(from),(rows)*(cols)*sizeof(float),cudaMemcpyHostToDevice))

static cublasHandle_t CB;
static curandGenerator_t RNG;
const static float zero = 0.0;
const static float one = 1.0;
const static float minusone = -1.0;

void print5(float *x);
static inline float *gpuArray(size_t nfloats);
static inline float *gpuCopy(size_t nfloats, float *cpuArray);
static inline float *gpuFill(size_t nfloats, float val);
#ifndef DBGMEM
#define gpuFree(x) { CUDA(cudaFree(x)); (x)=NULL; }
#else
static inline void gpuFreeDBG(void *x);
#define gpuFree(x) { gpuFreeDBG(x); (x)=NULL; }
#endif

__global__ void _reluforw(int n, float *y);
__global__ void _reluback(int n, float *y, float *dy);
__global__ void _logpforw(int nrows, int ncols, float *y);
__global__ void _softback(int nrows, int ncols, float *y, float *dy);
__global__ void _logploss(int nrows, int ncols, float *y, float *dy);
__global__ void _l1reg(int n, float l1, float *w, float *dw);
__global__ void _adagrad(int n, float eps, float *dw2, float *dw);
__global__ void _fill(int n, float val, float *x);
__global__ void _drop(int n, float *x, float *xmask, float dropout, float scale);
__global__ void _badd(int nrows, int ncols, float *y, float *b);
__global__ void _add1(int n, float val, float *x);

#define KCALL(f,...) {f<<<BLK,THR>>>(__VA_ARGS__); CUDA(cudaGetLastError()); }
void reluforw(int n, float *y) KCALL(_reluforw,n,y);
void reluback(int n, float *y, float *dy) KCALL(_reluback,n,y,dy);
void logpforw(int nrows, int ncols, float *y) KCALL(_logpforw,nrows,ncols,y);
void softback(int nrows, int ncols, float *y, float *dy) KCALL(_softback,nrows,ncols,y,dy);
void logploss(int nrows, int ncols, float *y, float *dy) KCALL(_logploss,nrows,ncols,y,dy);
void l1reg(int n, float l1, float *w, float *dw) KCALL(_l1reg,n,l1,w,dw);
void adagrad(int n, float eps, float *dw2, float *dw) KCALL(_adagrad,n,eps,dw2,dw);
void fill(int n, float val, float *x) KCALL(_fill,n,val,x);
void add1(int n, float val, float *x) KCALL(_add1,n,val,x);
void drop(int n, float *x, float *xmask, float dropout, float scale) KCALL(_drop,n,x,xmask,dropout,scale);
void badd(int nrows, int ncols, float *y, float *b) KCALL(_badd,nrows,ncols,y,b);

static inline void xforw(Layer l);
static inline void yforw(Layer l);
static inline void xback(Layer l);
static inline void yback(Layer l);
static inline float *tforw(Layer l, float *x, int xcols);


/* layer() constructs a layer with a weight matrix w of size
   (wrows,wcols).  b is a bias vector of length (wrows) or NULL if no
   bias is to be used.  Yfunc type determines the activation function,
   e.g. relu.  Xfunc determines preprocessing, if any, e.g. dropout.
   A layer has alloc/free responsibility of all its fields except x
   and dy, which it takes as input during forw and back respectively.
   Note that calls to layer functions may overwrite their inputs x and
   dy.
 */
Layer layer(Xfunc xfunc, Yfunc yfunc, int wrows, int wcols, float *w, float *b) {
  Layer l = (Layer) calloc(1, sizeof(struct LayerS));
  assert(l != NULL);
  l->xfunc = xfunc;
  l->yfunc = yfunc;
  l->wrows = wrows;
  l->wcols = wcols;
  l->learningRate = DEFAULT_LEARNING_RATE;
  if (w != NULL) l->w = gpuCopy(wrows*wcols, w);
  if (b != NULL) l->b = gpuCopy(wrows, b);
  return(l);
}

void lfree(Layer l) {
  gpuFree(l->w);
  gpuFree(l->b);
  //gpuFree(l->x);   // taken as input, not alloced
  gpuFree(l->y);   
  gpuFree(l->xones);
  gpuFree(l->xmask);

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

void set_adagrad(Layer l, float a) { l->adagrad = a; }
void set_nesterov(Layer l, float n) { l->nesterov = n; }
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
  assert(w != NULL);
  return layer(DROP, RELU, wrows, wcols, w, b);
}

Layer soft(int wrows, int wcols, float *w, float *b) {
  assert(w != NULL);
  return layer(DROP, SOFT, wrows, wcols, w, b);
}

static inline float *tforw(Layer l, float *x, int xcols) {
  // To prevent dropout during testing
  // Need to reconsider when other xfuncs implemented.
  Xfunc xf_save = l->xfunc;
  l->xfunc = NOXF;
  float *y = lforw(l, x, xcols);
  l->xfunc = xf_save;
  return y;
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
      gptr = tforw(net[l], gptr, batch);
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
    for (int l = 0; l < nlayer; l++) {
      gptr = lforw(net[l], gptr, batch); 
    }
    gpuSetMatrix(yrows, batch, &y[b*yrows], ygpu);
    gptr = ygpu; 
    for (int l = nlayer - 1; l >= 0; l--) {
      gptr = lback(net[l], gptr, (l>0));
    }
  }
  gpuFree(xgpu);
  gpuFree(ygpu);
}


float *lforw(Layer l, float *x, int xcols) {
  if (CB == NULL) CUBLAS(cublasCreate(&CB));

  // We assume x is already a device pointer.
  // Otherwise we'd have to do unnecessary copying between layers.
  // We assume xrows == l->wcols and x is column-major.  
  if (l->acols < xcols) { /* reallocate if columns changed */
    lclean(l);
    l->acols = xcols;
  }
  l->x = x;			/* we point to x, we do not copy */
  l->xcols = xcols;

  // x = f(x) where f is dropout etc.
  xforw(l);			/* dropout or other processing, overwrites x! */
  // y = w * x
  // gemm(opA,opB,m,n,k,α,A(m,k),lda=m,B(k,n),ldb=k,β,C(m,n),ldc=m): C = α op(A) op(B) + β C
  if (l->y == NULL) l->y = gpuArray(l->wrows * l->acols);
  CUBLAS(cublasSgemm(CB, CUBLAS_OP_N, CUBLAS_OP_N, l->wrows, l->xcols, l->wcols, &one, l->w, l->wrows, l->x, l->wcols, &zero, l->y, l->wrows));
  // y = y + b  with singleton expansion
  if (l->b != NULL) {
    // ger(m,n,α,x(m),incx=1,y(n),incy=1,A(m,n),lda=m): A = α x y' + A
    /*
    if (l->xones == NULL) l->xones = gpuFill(l->acols, 1.0);
    CUBLAS(cublasSger(CB, l->wrows, l->xcols, &one, l->b, 1, l->xones, 1, l->y, l->wrows));
    */
    badd(l->wrows, l->xcols, l->y, l->b);
  }
  // y = f(y) where f is relu, sigm etc.
  yforw(l);
  return l->y;
}

static inline void xforw(Layer l) {
  switch(l->xfunc) {
  case NOXF: break;
  case DROP:
    if (l->dropout > 0) {
      if (l->xmask == NULL) l->xmask = gpuArray(l->wcols * l->acols);
      randfill(l->wcols * l->xcols, l->xmask);
      KCALL(_drop,l->wcols * l->xcols, l->x, l->xmask, l->dropout, 1/(1-l->dropout));
    }
    break;
  default: assert("xforw != DROP not implemented yet"==NULL);
  }
}

static inline void yforw(Layer l) {
  switch(l->yfunc) {
  case NOYF: break;
  case RELU:
    _reluforw<<<BLK,THR>>>(l->wrows * l->xcols, l->y);
    CUDA(cudaGetLastError());
    break;
  case SOFT: break;
  default: assert("yforw != SOFT,RELU not implemented yet"==NULL);
  }
}

float *lback(Layer l, float *dy, int return_dx) {
  if (CB == NULL) CUBLAS(cublasCreate(&CB));

  // We assume dy is already a device pointer.
  // Otherwise we'd have to do unnecessary copying between layers.
  // We assume dy has the same size as l.y: (wrows,xcols)
  l->dy = dy;

  // dy = yback(dy) where yback is the derivative of yforw
  yback(l);

  // dw = dy * x'
  // gemm(opA,opB,m,n,k,α,A(m,k),lda=m,B(k,n),ldb=k,β,C(m,n),ldc=m): C = α op(A) op(B) + β C
  // m = wrows; n = wcols; k = xcols
  if (l->dw == NULL) l->dw = gpuArray(l->wrows * l->wcols);
  CUBLAS(cublasSgemm(CB, CUBLAS_OP_N, CUBLAS_OP_T, l->wrows, l->wcols, l->xcols, &one, l->dy, l->wrows, l->x, l->wcols, &zero, l->dw, l->wrows));

  if (l->b != NULL) {
    // db = sum(dy,2) = dy * ones
    // gemv(op,m,n,α,A(m,n),lda=m,x(n),incx=1,β,y(m),incy=1): y = α op(A) x + β y
    if (l->db == NULL) l->db = gpuArray(l->wrows);
    if (l->xones == NULL) l->xones = gpuFill(l->acols, 1.0);
    CUBLAS(cublasSgemv(CB, CUBLAS_OP_N, l->wrows, l->xcols, &one, l->dy, l->wrows, l->xones, 1, &zero, l->db, 1));
  }
  if (return_dx) { // dx is optional because it is expensive and unnecessary for input layer
    // dx=w' * dy
    // gemm(opA,opB,m,n,k,α,A(m,k),lda=m,B(k,n),ldb=k,β,C(m,n),ldc=m): C = α op(A) op(B) + β C
    // m = wcols, n = xcols, k = wrows
    if (l->dx == NULL) l->dx = gpuArray(l->wcols * l->acols);
    CUBLAS(cublasSgemm(CB, CUBLAS_OP_T, CUBLAS_OP_N, l->wcols, l->xcols, l->wrows, &one, l->w, l->wrows, l->dy, l->wrows, &zero, l->dx, l->wcols));
    xback(l);
    return l->dx;
  } else {
    return NULL;
  }
}

static inline void yback(Layer l) {
  switch(l->yfunc) {
  case NOYF: break;
  case RELU:
    _reluback<<<BLK,THR>>>(l->wrows * l->xcols, l->y, l->dy);
    CUDA(cudaGetLastError());
    break;
  case SOFT:
    _softback<<<BLK,THR>>>(l->wrows, l->xcols, l->y, l->dy);
    CUDA(cudaGetLastError());
    break;
  default: assert("yback != RELU,SOFT not implemented yet"==NULL);
  }
}

static inline void xback(Layer l) {
  switch(l->xfunc) {
  case NOXF: break;
  case DROP:
    if (l->dropout > 0) 
      KCALL(_drop, l->wcols * l->xcols, l->dx, l->xmask, l->dropout, 1/(1-l->dropout));
    break;
  default: assert("xforw != DROP not implemented yet"==NULL);
  }
}

void train(Layer *net, float *x, float *y, int nlayer, int xcols, int batch) {
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
    for (int l = 0; l < nlayer; l++)
      lupdate(net[l]);
  }
  gpuFree(xgpu);
  gpuFree(ygpu);
}


void lupdate(Layer l) {
  if (l->learningRate == 0) return;
  if (CB == NULL) CUBLAS(cublasCreate(&CB));
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
    CUBLAS(cublasSaxpy(CB, nw, &l->L2, l->w, 1, l->dw, 1));
  }
  if (l->adagrad > 0) {
    /* ADAGRAD:
       dw2 += dw.*dw 
       dw /= (epsilon + sqrt(dw2))
       and similarly for db.
    */
    if (l->dw2 == NULL) l->dw2 = gpuFill(nw, 0.0);
    _adagrad<<<BLK,THR>>>(nw, l->adagrad, l->dw2, l->dw);
    if (nb) { 
      if (l->db2 == NULL) l->db2 = gpuFill(nb, 0.0);
      _adagrad<<<BLK,THR>>>(nb, l->adagrad, l->db2, l->db); 
    }
    CUDA(cudaGetLastError());
  }
  if (l->learningRate != 1) {
    /* LearningRate:
       Scale dw and db with the learning rate.
       dw,db *= learningRate
       scal(n,α,x(n),incx=1): x = α x
    */
    CUBLAS(cublasSscal(CB, nw, &l->learningRate, l->dw, 1));
    if (nb) CUBLAS(cublasSscal(CB, nb, &l->learningRate, l->db, 1));
  }
  if (l->momentum != 0 || l->nesterov != 0) {  
    /* Momentum:
       why do we apply it here?
       do we apply it to db?
       check the following:
       dw1 = momentum * dw1 + dw
       dw = dw1   :without nesterov
       dw = momentum * dw1 + dw   :with nesterov
       TODO: clean this up a bit...
    */
    assert(l->momentum == 0 || l->nesterov == 0);
    float m = l->momentum != 0 ? l->momentum : l->nesterov;
    if (l->dw1 == NULL) l->dw1 = gpuFill(nw, 0.0);
    CUBLAS(cublasSscal(CB, nw, &m, l->dw1, 1));
    CUBLAS(cublasSaxpy(CB, nw, &one, l->dw, 1, l->dw1, 1));
    if (l->nesterov != 0) {
      CUBLAS(cublasSaxpy(CB, nw, &m, l->dw1, 1, l->dw, 1));
    } else {
      CUBLAS(cublasScopy(CB, nw, l->dw1, 1, l->dw, 1));
    }
    if (nb) {
      if (l->db1 == NULL) l->db1 = gpuFill(nb, 0.0);
      CUBLAS(cublasSscal(CB, nb, &m, l->db1, 1));
      CUBLAS(cublasSaxpy(CB, nb, &one, l->db, 1, l->db1, 1));
      if (l->nesterov != 0) {
	CUBLAS(cublasSaxpy(CB, nb, &m, l->db1, 1, l->db, 1));
      } else {
	CUBLAS(cublasScopy(CB, nb, l->db1, 1, l->db, 1));
      }
    }
  }
  /* Finally apply gradient descent: w -= dw, b -= db */
  CUBLAS(cublasSaxpy(CB, nw, &minusone, l->dw, 1, l->w, 1));
  if (nb) CUBLAS(cublasSaxpy(CB, nb, &minusone, l->db, 1, l->b, 1));

  if (l->maxnorm != 0) {
    /* MaxNorm:

     */
    assert(1 == 0);  // TBD
  }
}

void lclean(Layer l) {
  /* Get rid of all arrays which depend on input xcols */
  gpuFree(l->y);
  gpuFree(l->xones);
  gpuFree(l->dx);
  gpuFree(l->xmask);
}

__global__ void _fill(int n, float val, float *x) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[i] = val;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _add1(int n, float val, float *x) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[i] += val;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _badd(int nrows, int ncols, float *y, float *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int n = nrows * ncols;
  while (i < n) {
    y[i] += b[i % nrows];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _drop(int n, float *x, float *xmask, float dropout, float scale) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (xmask[i] < dropout) x[i] = 0;
    else x[i] *= scale;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _adagrad(int n, float eps, float *dw2, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    dw2[i] += dw[i] * dw[i];
    dw[i] /= (eps + sqrt(dw2[i]));
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _l1reg(int n, float l1, float *w, float *dw) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (w[i] > 0) dw[i] += l1;
    else if (w[i] < 0) dw[i] -= l1;
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
  /* y is layer output, i.e. unnormalized log probabilities.
       On output y will contain normalized probabilities.
       Conceptually this is a forward calculation but we do it here for efficiency.
     dy is the label matrix: each column is a one-hot vector indicating the correct label.
       On output dy will be the gradient of softmax loss wrt probabilities.
   */
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

__global__ void _logploss(int nrows, int ncols, float *y, float *dy) {
  /* Similar to softmaxloss, except y is assumed normalized logp and is not overwritten.
     y is layer output, i.e. normalized log probabilities.
     dy is the label matrix: each column is a one-hot vector indicating the correct label.
     On output dy will be the gradient of softmax loss wrt log probabilities.
   */
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int i0, i1;
  while (col < ncols) {
    i0 = col * nrows;
    i1 = i0  + nrows;
    for (int i=i0; i<i1; i++) {
      dy[i] = (exp(y[i]) - dy[i]) / ncols;
    }
    col += blockDim.x * gridDim.x;
  }
}

__global__ void _logpforw(int nrows, int ncols, float *y) {
  /* y is layer output, i.e. unnormalized log probabilities.
     On output y will contain normalized probabilities.
  */
  float ymax, z, logz;
  int i0, i1;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  while (col < ncols) {
    i0 = col * nrows;
    i1 = i0  + nrows;
    ymax = -INFINITY;
    for (int i=i0; i<i1; i++) {
      if (y[i] > ymax) {
	ymax = y[i];
      }
    }
    z = 0;
    for (int i=i0; i<i1; i++) {
      y[i] -= ymax;
      z += exp(y[i]);
    }
    logz = log(z);
    for (int i=i0; i<i1; i++) {
      y[i] -= logz;
    }
    col += blockDim.x * gridDim.x;
  }
}

#ifndef DBGMEM

static inline float *gpuArray(size_t nfloats) {
  float *gptr;
  CUDA(cudaMalloc((void **) &gptr, nfloats * sizeof(float)));
  return gptr;
}

#else

static inline float *gpuArray(size_t nfloats) {
  size_t mfree1, mtotal1, mfree2, mtotal2;
  float *gptr;
  CUDA(cudaMemGetInfo(&mfree1, &mtotal1));
  CUDA(cudaMalloc((void **) &gptr, nfloats * sizeof(float)));
  CUDA(cudaMemGetInfo(&mfree2, &mtotal2));
  assert(mtotal1 == mtotal2);
  fprintf(stderr, "gpuArray: f%zu = %zu =? %zd = %zu - %zu (%zu)\n", 
	  nfloats, 4*nfloats, mfree1-mfree2, mfree1, mfree2, mtotal1);
  return gptr;
}

static inline void gpuFreeDBG(void *x) { 
  if (x == NULL) return;
  size_t mfree1, mtotal1, mfree2, mtotal2;
  CUDA(cudaMemGetInfo(&mfree1, &mtotal1));
  CUDA(cudaFree(x)); 
  CUDA(cudaMemGetInfo(&mfree2, &mtotal2));
  assert(mtotal1 == mtotal2);
  fprintf(stderr, "gpuFree: %zu = %zu - %zu (%zu)\n",
	  mfree2-mfree1, mfree2, mfree1, mtotal1);
}
#endif

static inline float *gpuCopy(size_t nfloats, float *cpuArray) {
  float *gptr = gpuArray(nfloats);
  CUDA(cudaMemcpy(gptr, cpuArray, nfloats * sizeof(float), cudaMemcpyHostToDevice));
  return gptr;
}

static inline float *gpuFill(size_t nfloats, float val) {
  float *gptr = gpuArray(nfloats);
  _fill<<<BLK,THR>>>(nfloats, val, gptr);
  return gptr;
}

void gpuseed(unsigned long long seed) {
  CURAND(curandCreateGenerator(&RNG, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND(curandSetPseudoRandomGeneratorSeed(RNG, seed));
}

void randfill(int n, float *x) {
  if (RNG == NULL) CURAND(curandCreateGenerator(&RNG, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND(curandGenerateUniform(RNG, x, n));
}

/* Alternative cublas implementations of broadcasting and column sum */
#define ONES 1000000
static float *ones;

void badd_cublas(int nrows, int ncols, float *y, float *b) {
  if (CB == NULL) CUBLAS(cublasCreate(&CB));
  if (ones == NULL) ones = gpuFill(ONES, 1.0);
  assert(ncols < ONES);
  CUBLAS(cublasSger(CB, nrows, ncols, &one, b, 1, ones, 1, y, nrows));
}

void bsum(int nrows, int ncols, float *y, float *b) {
  if (CB == NULL) CUBLAS(cublasCreate(&CB));
  if (ones == NULL) ones = gpuFill(ONES, 1.0);
  assert(ncols < ONES);
  CUBLAS(cublasSgemv(CB, CUBLAS_OP_N, nrows, ncols, &one, y, nrows, ones, 1, &zero, b, 1));
}

/* Debugging */

void print5(float *x) {
  float xx[5];
  CUDA(cudaMemcpy(xx,x,5*sizeof(float),cudaMemcpyDeviceToHost));
  for (int i=0; i<5; i++) printf("%g ", xx[i]);
  printf("...\n");
}

