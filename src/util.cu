#include "kunet.h"

/* TODO: These should be directly called from julia. */

curandGenerator_t RNG;

extern "C" {

  void gpuseed(unsigned long long seed) {
    assert(curandCreateGenerator(&RNG, CURAND_RNG_PSEUDO_DEFAULT)==CURAND_STATUS_SUCCESS);
    assert(curandSetPseudoRandomGeneratorSeed(RNG, seed)==CURAND_STATUS_SUCCESS);
  }

  void randfill32(int n, float *x) CURAND(curandGenerateUniform(RNG, x, n));
  void randfill64(int n, double *x) CURAND(curandGenerateUniformDouble(RNG, x, n));
  void initgaussian32(float *x, int n, float mean, float std) CURAND(curandGenerateNormal(RNG, x, n, mean, std));
  void initgaussian64(double *x, int n, double mean, double std) CURAND(curandGenerateNormalDouble(RNG, x, n, mean, std));
  void *rng() { return RNG; }
}
