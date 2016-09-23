#ifndef _KNET_H
#define _KNET_H
#include <cuda_runtime.h>
#include <assert.h>
#define CUDA(_s) assert((_s) == cudaSuccess)
#define BLK 128
#define THR 128
#define KCALL(f,...) {f<<<BLK,THR>>>(__VA_ARGS__); CUDA(cudaGetLastError()); }

// renaming based on http://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error
static __device__ double molecularAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(val +
					 __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)} while (assumed != old);
  } while(assumed != old); 
  return __longlong_as_double(old);
}

static __device__ float molecularAdd(float* address, float val)
{
  return atomicAdd(address, val);
}

#endif
