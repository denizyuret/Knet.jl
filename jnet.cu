// nvcc --shared --compiler-options -fPIC -o libjnet.so jnet.cu -lcublas

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include "jnet.h"
#define BLK 128
#define THR 128

typedef enum { NOOP, RELU, SOFT } LayerType;

typedef struct LayerS {
  LayerType type; // type of activation function	
  float *w;	// weight matrix (wrows,wcols)
  float *b;	// bias vector (wrows)

  float *dw;	// gradient wrt weight matrix
  float *dw1;	// moving average of gradients for momentum
  float *dw2;	// sum of squared gradients for adagrad

  float *db;	// gradient wrt bias vector
  float *db1;	// moving average of gradients for momentum
  float *db2;	// sum of squared gradients for adagrad

  float *x;	// last input (wcols,xcols)
  float *y;	// last output (wrows,xcols)
  float *dx;	// gradient wrt input
  float *dy;	// gradient wrt output
  float *xmask;	// input mask for dropout
  float *xones;	// vector of ones for bias calculation (xcols)

  int wrows;
  int wcols;
  int xcols;
} *Layer;


Layer layer(LayerType type, int wrows, int wcols, float *w, float *b);
float *forw(Layer l, float *x, int xcols);
float *initforw(Layer l, float *x, int xcols);

__global__ void _fill(float *y, int n, float val);
__global__ void _reluforw(float *y, int n);
__global__ void _reluback(float *dy, float *y, int n);
__global__ void _softback(float *dy, float *y, int nrows, int ncols);


#define CUDA(s) {\
    cudaError_t err = (s);\
    if (err != cudaSuccess) {\
      fprintf(stderr, "cuda error: %s\n", cudaGetErrorString(err));\
      exit(EXIT_FAILURE);\
    }\
  }


#define NOTNULL(s) {\
    if ((s) == NULL) {\
      fprintf(stderr, "Unexpected null value\n");\
      exit(EXIT_FAILURE);\
    }\
  }


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
  if (l->b != NULL) CUDA(cudaFree(l->b));
  free(l);
}

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
  l->xcols = xcols;
  return x;
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

