#ifndef _KUNET_H
#define _KUNET_H
#include <cuda_runtime.h>
#include <assert.h>
#define CUDA(_s) assert((_s) == cudaSuccess)
#define BLK 128
#define THR 128
#define KCALL(f,...) {f<<<BLK,THR>>>(__VA_ARGS__); CUDA(cudaGetLastError()); }
#include <curand.h>
extern curandGenerator_t RNG;
#define CURAND(_s) {							\
    if (RNG==NULL) assert(curandCreateGenerator(&RNG, CURAND_RNG_PSEUDO_DEFAULT)==CURAND_STATUS_SUCCESS); \
    assert((_s) == CURAND_STATUS_SUCCESS);				\
  }
#endif
