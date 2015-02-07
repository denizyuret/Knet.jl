#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "jnet.h"

void h5read(hid_t file_id, char *name, float **data, hsize_t *dims) {
  H5LTget_dataset_info(file_id,name,dims,NULL,NULL);
  *data = calloc(dims[0]*dims[1], sizeof(float));
  H5LTread_dataset_float(file_id,name,*data);
}

void forwtest(Layer *net, float *x, float *y, int nlayer, int xcols, int batch) {
  fprintf(stderr, "Calling forward... ");
  clock_t t0 = clock();
  forward(net, x, y, nlayer, xcols, batch);
  clock_t t1 = clock();
  fprintf(stderr, "%g seconds\n", (double)(t1-t0)/CLOCKS_PER_SEC);
}

void forwbacktest(Layer *net, float *x, float *y, int nlayer, int xcols, int batch) {
  fprintf(stderr, "Calling forwback... ");
  clock_t t0 = clock();
  forwback(net, x, y, nlayer, xcols, batch);
  clock_t t1 = clock();
  fprintf(stderr, "%g seconds\n", (double)(t1-t0)/CLOCKS_PER_SEC);
}

int main(void) {
  float *x, *y, *ygold, *yorig, *w1, *b1, *w2, *b2;
  int xcols, wrows, wcols, yrows;
  hsize_t dims[2];
  hid_t file_id = H5Fopen("dev.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
  h5read(file_id, "/w1", &w1, dims);
  wrows = dims[1]; wcols = dims[0];
  h5read(file_id, "/x", &x, dims);
  assert(dims[1] == wcols); xcols = dims[0];
  h5read(file_id, "/y", &yorig, dims); 
  yrows = dims[1]; assert(dims[0] == xcols);
  h5read(file_id, "/ygold", &ygold, dims); 
  assert(dims[1] == yrows); assert(dims[0] == xcols);
  h5read(file_id, "/w2", &w2, dims); 
  assert(dims[1] == yrows); assert(dims[0] == wrows);
  h5read(file_id, "/b1", &b1, dims);
  assert(dims[1] == wrows); assert(dims[0] == 1);
  h5read(file_id, "/b2", &b2, dims); 
  assert(dims[1] == yrows); assert(dims[0] == 1);
  H5Fclose(file_id);
  printf("%g %g %g\n", yorig[0], yorig[1], yorig[2]);

  y = calloc(yrows * xcols, sizeof(float));
  int batch = 1000;
  Layer net[2];
  net[0] = relu(wrows, wcols, w1, b1);
  net[1] = soft(yrows, wrows, w2, b2);
  
  forwtest(net, x, y, 2, xcols, batch);
  printf("%g %g %g\n", y[0], y[1], y[2]);
  float maxdiff = 0; int i;
  for (i = yrows * xcols - 1; i >= 0; i--) {
    float d = y[i]-yorig[i];
    if (d < 0) d = -d;
    if (d > maxdiff) maxdiff = d;
  }
  printf("maxdiff=%g\n", maxdiff);
  forwtest(net, x, y, 2, xcols, batch);
  forwtest(net, x, y, 2, xcols, batch);

  forwbacktest(net, x, y, 2, xcols, batch);
  forwbacktest(net, x, y, 2, xcols, batch);
  forwbacktest(net, x, y, 2, xcols, batch);

  lfree(net[0]);
  lfree(net[1]);
}

