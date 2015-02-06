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

typedef enum { NOOP, RELU, SOFT } LayerType;

typedef struct LayerS {
  int wrows, wcols, xcols; // size params

  LayerType type;	// type of activation function	
  float *w;		// weight matrix (wrows,wcols)
  float *b;		// bias vector (wrows)
  float *x;		// last input (wcols,xcols)
  float *y;		// last output (wrows,xcols)
  float *xones;		// vector of ones for bias calculation (xcols)
  float *xmask;		// input mask for dropout

  float *dw;		// gradient wrt weight matrix
  float *db;		// gradient wrt bias vector
  float *dx;		// gradient wrt input
  float *dy;		// gradient wrt output

  float *dw1;		// moving average of gradients for momentum
  float *dw2;		// sum of squared gradients for adagrad
  float *db1;		// moving average of gradients for momentum
  float *db2;		// sum of squared gradients for adagrad

  bool adagrad;		// adagrad during weight updates
  bool nesterov;	// nesterov during weight updates
  float learningRate;	// default=0, acts like 1.0
  float momentum;	// default=0
  float dropout;	// probability of dropping inputs, default=0
  float maxnorm;	// default=0, acts like inf
  float L1, L2;		// L1,L2 regularization, default=0

} *Layer;


Layer layer(LayerType type, int wrows, int wcols, float *w, float *b);
float *forw(Layer l, float *x, int xcols);
float *initforw(Layer l, float *x, int xcols);
float *back(Layer l, float *dy, bool dx);
float *initback(Layer l, float *dy, bool dx);
void initupdate(Layer l);

__global__ void _fill(float *y, int n, float val);
__global__ void _reluforw(float *y, int n);
__global__ void _reluback(float *dy, float *y, int n);
__global__ void _softback(float *dy, float *y, int nrows, int ncols);


#define CUDA(_s) assert((_s) == cudaSuccess)
#define NOTNULL(_s) assert((_s) != NULL)


Layer layer(LayerType type, int wrows, int wcols, float *w, float *b) {
  Layer l;
  NOTNULL(w);
  NOTNULL(l = (Layer) calloc(1, sizeof(struct LayerS)));
  l = (Layer) calloc(1, sizeof(struct LayerS));
  l->type = type;
  l->wrows = wrows;
  l->wcols = wcols;
  int wsize = wrows*wcols*sizeof(float);
  CUDA(cudaMalloc((void **) &l->w, wsize));
  CUDA(cudaMemcpy(l->w, w, wsize, cudaMemcpyHostToDevice));
  if (b != NULL) {
    int bsize = wrows*sizeof(float);
    CUDA(cudaMalloc((void **) &l->b, bsize));
    CUDA(cudaMemcpy(l->b, b, bsize, cudaMemcpyHostToDevice));
  }
  return(l);
}

extern "C" void lfree(Layer l) {
  CUDA(cudaFree(l->w));
  CUDA(cudaFree(l->b));
  //CUDA(cudaFree(l->x));   // taken as input, not alloced
  CUDA(cudaFree(l->y));   
  CUDA(cudaFree(l->xones));
  //CUDA(cudaFree(l->xmask));

  CUDA(cudaFree(l->dw));
  CUDA(cudaFree(l->db));
  CUDA(cudaFree(l->dx));
  //CUDA(cudaFree(l->dy));  // taken as input, not alloced

  CUDA(cudaFree(l->dw1));
  CUDA(cudaFree(l->dw2));
  CUDA(cudaFree(l->db1));
  CUDA(cudaFree(l->db2));
  free(l);
}

extern "C" void adagrad(Layer l, int i) { l->adagrad = i; }
extern "C" void nesterov(Layer l, int i) { l->nesterov = i; }
extern "C" void learningRate(Layer l, float lr) { l->learningRate = lr; }
extern "C" void momentum(Layer l, float m) { l->momentum = m; }
extern "C" void dropout(Layer l, float d) { l->dropout = d; }
extern "C" void maxnorm(Layer l, float m) { l->maxnorm = m; }
extern "C" void L1(Layer l, float m) { l->L1 = m; }
extern "C" void L2(Layer l, float m) { l->L2 = m; }

extern "C" int lsize(Layer l, int i) { 
  switch(i) {
  case 1: return l->wrows;
  case 2: return l->wcols;
  default: fprintf(stderr, "size argument must be 1 or 2\n");
    exit(EXIT_FAILURE);
  }
}

extern "C" Layer relu(int wrows, int wcols, float *w, float *b) {
  return layer(RELU, wrows, wcols, w, b);
}

extern "C" Layer soft(int wrows, int wcols, float *w, float *b) {
  return layer(SOFT, wrows, wcols, w, b);
}

extern "C" void forward(Layer *net, float *x, float *y, int nlayer, int xcols, int batch) {
  float *xgpu;
  int xrows = net[0]->wcols;
  int yrows = net[nlayer-1]->wrows;
  int xsize = xrows * batch * sizeof(float);
  int ysize = yrows * batch * sizeof(float);
  CUDA(cudaMalloc((void **) &xgpu, xsize));
  for (int b = 0; b < xcols; b += batch) {
    if (b + batch > xcols) {
      batch = xcols - b;
      xsize = xrows * batch * sizeof(float);
      ysize = yrows * batch * sizeof(float);
    }
    CUDA(cudaMemcpy(xgpu, &x[b * xrows], xsize, cudaMemcpyHostToDevice));
    float *ygpu = xgpu;
    for (int l = 0; l < nlayer; l++) {
      ygpu = forw(net[l], ygpu, batch);
    }
    CUDA(cudaMemcpy(&y[b * yrows], ygpu, ysize, cudaMemcpyDeviceToHost));
  }
  CUDA(cudaFree(xgpu));
}

extern "C" void forwback(Layer *net, float *x, float *y, int nlayer, int xcols, int batch) {
  float *xgpu, *ygpu;
  int xrows = net[0]->wcols;
  int yrows = net[nlayer-1]->wrows;
  int xsize = xrows * batch * sizeof(float);
  int ysize = yrows * batch * sizeof(float);
  CUDA(cudaMalloc((void **) &xgpu, xsize));
  CUDA(cudaMalloc((void **) &ygpu, ysize));
  for (int b = 0; b < xcols; b += batch) {
    if (b + batch > xcols) {
      batch = xcols - b;
      xsize = xrows * batch * sizeof(float);
      ysize = yrows * batch * sizeof(float);
    }
    CUDA(cudaMemcpy(xgpu, &x[b * xrows], xsize, cudaMemcpyHostToDevice));
    float *xptr = xgpu;
    for (int l = 0; l < nlayer; l++)
      xptr = forw(net[l], xptr, batch);
    CUDA(cudaMemcpy(ygpu, &y[b * yrows], ysize, cudaMemcpyHostToDevice));
    float *yptr = ygpu;
    for (int l = nlayer - 1; l >= 0; l--)
      yptr = back(net[l], yptr, (l>0));
  }
  CUDA(cudaFree(xgpu));
  CUDA(cudaFree(ygpu));
}


float *forw(Layer l, float *x, int xcols) {
  // We assume x is already a device pointer.
  // Otherwise we'd have to do unnecessary copying between layers.
  // We assume xrows == l->wcols and x is column-major.  
  l->x = initforw(l, x, xcols);

  // l->y = l->w * l->x
  // gemm(opA,opB,m,n,k,α,A(m,k),lda=m,B(k,n),ldb=k,β,C(m,n),ldc=m): C = α op(A) op(B) + β C
  cublasSgemm('N', 'N', l->wrows, l->xcols, l->wcols, 1.0, l->w, l->wrows, l->x, l->wcols, 0.0, l->y, l->wrows);

  if (l->b != NULL) {
    // l->y = l->y + l->b with singleton expansion
    // ger(m,n,α,x(m),incx=1,y(n),incy=1,A(m,n),lda=m): A = α x y' + A
    cublasSger(l->wrows, l->xcols, 1.0, l->b, 1, l->xones, 1, l->y, l->wrows);
  }
  // l->y = fforw(l->y)
  switch(l->type) {
  case RELU:
    _reluforw<<<BLK,THR>>>(l->y, l->wrows * l->xcols);
    CUDA(cudaGetLastError());
    break;
  }
  return(l->y);
}

float *initforw(Layer l, float *x, int xcols) {
  // Alloc/realloc l->y and l->xones if necessary
  // Update l->xcols
  int yrows = l->wrows;
  int ycols = xcols;
  if ((l->y == NULL) || (l->xcols != xcols)) {
    CUDA(cudaFree(l->y));
    CUDA(cudaMalloc((void **) &l->y, yrows * ycols * sizeof(float)));
  }
  if ((l->b != NULL) && ((l->xones == NULL) || l->xcols != xcols)) {
    CUDA(cudaFree(l->xones));
    CUDA(cudaMalloc((void **) &l->xones, xcols * sizeof(float)));
    _fill<<<BLK,THR>>>(l->xones, xcols, 1.0);
    CUDA(cudaGetLastError());
  }
  if ((l->dx != NULL) && (l->xcols != xcols)) {
    CUDA(cudaFree(l->dx));
    l->dx = NULL;
  }
  l->xcols = xcols;
  return x;
}

float *back(Layer l, float *dy, bool dx) {
  // We assume dy is already a device pointer.
  // Otherwise we'd have to do unnecessary copying between layers.
  // We assume dy has the same size as l.y: (wrows,xcols)
  l->dy = initback(l, dy, dx);
  
  // dy = fback(dy)
  switch(l->type) {
  case RELU:
    _reluback<<<BLK,THR>>>(l->dy, l->y, l->wrows * l->xcols);
    CUDA(cudaGetLastError());
    break;
  case SOFT:
    _softback<<<BLK,THR>>>(l->dy, l->y, l->wrows, l->xcols);
    CUDA(cudaGetLastError());
    break;
  }

  // dw = dy * x'
  // gemm(opA,opB,m,n,k,α,A(m,k),lda=m,B(k,n),ldb=k,β,C(m,n),ldc=m): C = α op(A) op(B) + β C
  // m = wrows; n = wcols; k = xcols
  cublasSgemm('N', 'T', l->wrows, l->wcols, l->xcols, 1.0, l->dy, l->wrows, l->x, l->wcols, 0.0, l->dw, l->wrows);

  if (l->b != NULL) {
    // db = sum(dy,2) = dy * ones
    // gemv(op,m,n,α,A(m,n),lda=m,x(n),incx=1,β,y(m),incy=1): y = α op(A) x + β y
    cublasSgemv('N', l->wrows, l->xcols, 1.0, l->dy, l->wrows, l->xones, 1, 0.0, l->db, 1);
  }
  if (dx) { // dx is optional because it is expensive and unnecessary for input layer
    // dx=w' * dy
    // gemm(opA,opB,m,n,k,α,A(m,k),lda=m,B(k,n),ldb=k,β,C(m,n),ldc=m): C = α op(A) op(B) + β C
    // m = wcols, n = xcols, k = wrows
    cublasSgemm('T', 'N', l->wcols, l->xcols, l->wrows, 1.0, l->w, l->wrows, l->dy, l->wrows, 0.0, l->dx, l->wcols);
  }
  return l->dx;
}

float *initback(Layer l, float *dy, bool dx) {
  if (l->dw == NULL)
    CUDA(cudaMalloc((void **) &l->dw, l->wrows * l->wcols * sizeof(float)));
  if ((l->b != NULL) && (l->db == NULL))
    CUDA(cudaMalloc((void **) &l->db, l->wrows * sizeof(float)));
  if (dx && (l->dx == NULL))
    CUDA(cudaMalloc((void **) &l->dx, l->wcols * l->xcols * sizeof(float)));
  return dy;
}

extern "C" void update(Layer l) {
  initupdate(l);
  int nw = l->wcols * l->wrows;
  int nb = (l->b == NULL ? 0 : l->wrows);
  if (l->L1 != 0) {
    assert(1 == 0); // TBD
  }
  if (l->L2 != 0) { // dw += L2 * w
    // axpy(n,α,x(n),incx=1,y(n),incy=1): y = α x + y
    cublasSaxpy(nw, l->L2, l->w, 1, l->dw, 1);
    // We do not apply L2 to l->b??
  }
  if (l->adagrad) { // dw2 += dw.*dw; dw /= (epsilon + sqrt(dw2))
    assert(1 == 0); // TBD
  }
  if (l->learningRate != 0) {  // dw *= learningRate
    // scal(n,α,x(n),incx=1): x = α x
    cublasSscal(nw, l->learningRate, l->dw, 1);
    if (nb) cublasSscal(nb, l->learningRate, l->db, 1);
  }
  if (l->momentum != 0) {  // dw1 = momentum * dw1 + dw
    // do we apply momentum to db?
    cublasSscal(nw, l->momentum, l->dw1, 1);
    cublasSaxpy(nw, 1.0, l->dw, 1, l->dw1, 1);
    if (l->nesterov) {	// TODO: check this!
      cublasSaxpy(nw, l->momentum, l->dw1, 1, l->dw, 1);
    } else {
      cublasScopy(nw, l->dw1, 1, l->dw, 1);
    }
  }
  cublasSaxpy(nw, -1.0, l->dw, 1, l->w, 1);  // w -= dw
  if (nb) cublasSaxpy(nb, -1.0, l->db, 1, l->b, 1);  // b -= db
  if (l->maxnorm != 0) {
    assert(1 == 0);  // TBD
  }
}

void initupdate(Layer l) {
  if (l->adagrad && (l->dw2 == NULL)) {
    CUDA(cudaMalloc((void **) &(l->dw2), l->wcols*l->wrows*sizeof(float)));
    _fill<<<BLK,THR>>>(l->dw2, l->wcols*l->wrows, 0.0);
    CUDA(cudaGetLastError());
    if (l->b != NULL) {
      assert(l->db2 == NULL);
      CUDA(cudaMalloc((void **) &(l->db2), l->wrows*sizeof(float)));
      _fill<<<BLK,THR>>>(l->db2, l->wrows, 0.0);
      CUDA(cudaGetLastError());
    }
  }
  if ((l->momentum != 0) && (l->dw1 == NULL)) {
    CUDA(cudaMalloc((void **) &(l->dw1), l->wcols*l->wrows*sizeof(float)));
    _fill<<<BLK,THR>>>(l->dw1, l->wcols*l->wrows, 0.0);
    CUDA(cudaGetLastError());
    if (l->b != NULL) {
      assert(l->db1 == NULL);
      CUDA(cudaMalloc((void **) &(l->db1), l->wrows*sizeof(float)));
      _fill<<<BLK,THR>>>(l->db1, l->wrows, 0.0);
      CUDA(cudaGetLastError());
    }
  }
}



__global__ void _fill(float *y, int n, float val)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = val;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _reluforw(float *y, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] < 0) y[i] = 0;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _reluback(float *dy, float *y, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] <= 0) dy[i] = 0;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _softback(float *dy, float *y, int nrows, int ncols)
{
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

