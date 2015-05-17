#include "kunet.h"
#include <curand.h>
#define CURAND(_s) assert((_s) == CURAND_STATUS_SUCCESS)

/* TODO: These should be directly called from julia. */

static curandGenerator_t RNG;

extern "C" {

void gpuseed(unsigned long long seed) {
  CURAND(curandCreateGenerator(&RNG, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND(curandSetPseudoRandomGeneratorSeed(RNG, seed));
}

void randfill32(int n, float *x) {
  if (RNG == NULL) CURAND(curandCreateGenerator(&RNG, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND(curandGenerateUniform(RNG, x, n));
}

void randfill64(int n, double *x) {
  if (RNG == NULL) CURAND(curandCreateGenerator(&RNG, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND(curandGenerateUniformDouble(RNG, x, n));
}

}
