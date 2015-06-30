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

}
