#ifndef _KNET_H
#define _KNET_H
#include <cuda_runtime.h>
#include <assert.h>
#define CUDA(_s) assert((_s) == cudaSuccess)
#define BLK 128
#define THR 128
#define KCALL(f,...) {f<<<BLK,THR>>>(__VA_ARGS__); CUDA(cudaGetLastError()); }
#endif
