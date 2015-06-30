#include "kunet.h"
#include <curand.h>
curandGenerator_t RNG;
#define CURAND(_s) {							\
    if (RNG==NULL) assert(curandCreateGenerator(&RNG, CURAND_RNG_PSEUDO_DEFAULT)==CURAND_STATUS_SUCCESS); \
    assert((_s) == CURAND_STATUS_SUCCESS);				\
  }

extern "C" {

  void *gpurng() { return RNG; }

  void gpuseed(unsigned long long seed) {
    // need to regenerate RNG for the seed to take effect
    assert(curandCreateGenerator(&RNG, CURAND_RNG_PSEUDO_DEFAULT)==CURAND_STATUS_SUCCESS);
    assert(curandSetPseudoRandomGeneratorSeed(RNG, seed)==CURAND_STATUS_SUCCESS);
  }

  void rand32(float* x, int n) CURAND(curandGenerateUniform(RNG, x, n));
  void rand64(double *x, int n) CURAND(curandGenerateUniformDouble(RNG, x, n));

  void randn32(float *x, int n, float mean, float std) {
    // this requires n to be even, so we'll fix it here
    if (n%2 == 0) {
      CURAND(curandGenerateNormal(RNG, x, n, mean, std));
    } else {
      CURAND(curandGenerateNormal(RNG, x, n, mean, std));
      CURAND(curandGenerateNormal(RNG, x, n, mean, std));
    }
  }
}
