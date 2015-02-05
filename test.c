#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "jnet.h"

float *randf(int n) {
  int i;
  float *x = (float*) malloc(n * sizeof(float));
  for (i=0; i < n; i++) {
    x[i] = (float)rand()/RAND_MAX;
  }
  return x;
}

void forwtest(Layer *net, float *x, float *y, int nlayer, int xcols, int batch) {
  fprintf(stderr, "Calling forward... ");
  clock_t t0 = clock();
  forward(net, x, y, nlayer, xcols, batch);
  clock_t t1 = clock();
  fprintf(stderr, "%g seconds\n", (double)(t1-t0)/CLOCKS_PER_SEC);
}

int main() {
  fprintf(stderr, "Initializing\n");
  int batch = 10000;
  int xrows = 1326;
  int xcols = 76834;
  int wrows = 20000;		/* wcols=xrows */
  int yrows = 3;		/* ycols=xcols */
  float *x = randf(xrows * xcols);
  float *y = randf(yrows * xcols);
  float *w1 = randf(wrows * xrows);
  float *b1 = randf(wrows);
  float *w2 = randf(yrows * wrows);
  float *b2 = randf(yrows);
  Layer net[2];
  net[0] = relu(wrows, xrows, w1, b1);
  net[1] = soft(yrows, wrows, w2, b2);
  
  forwtest(net, x, y, 2, xcols, batch);
  forwtest(net, x, y, 2, xcols, batch);
  forwtest(net, x, y, 2, xcols, batch);

  lfree(net[0]);
  lfree(net[1]);
}
