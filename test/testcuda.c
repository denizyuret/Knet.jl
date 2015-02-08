#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cuda.h>
#include "jnet.h"

void print3(const char *str, float *tmp) {
  printf("%s: %g %g %g\n", str, tmp[0], tmp[1], tmp[2]);
}

void gprint3(const char *str, float *gptr) {
  float tmp[3];
  assert(cuMemcpyDtoH(tmp, (CUdeviceptr) gptr, 12) == CUDA_SUCCESS);
  printf("%s: %g %g %g\n", str, tmp[0], tmp[1], tmp[2]);
}

float *gather(float *gptr, int n) {
  size_t bytes = n * sizeof(float);
  float *cptr = (float *) malloc(bytes);
  assert(cuMemcpyDtoH(cptr, (CUdeviceptr) gptr, bytes) == CUDA_SUCCESS);
  return cptr;
}

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

void maxdiff(const char *str, int n, float *x, float *y) {
  float mdiff = 0; int i;
  for (i = n - 1; i >= 0; i--) {
    float d = y[i]-x[i];
    if (d < 0) d = -d;
    if (d > mdiff) mdiff = d;
  }
  printf("maxdiff(%s)=%g\n", str, mdiff);
}

int main(void) {
  printf("Loading dev.h5\n");
  float *x, *ytest, *ygold, *yorig, *w1, *b1, *w2, *b2;
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
  // print3("orig y", yorig);

  int ysize = yrows * xcols;
  ytest = calloc(ysize, sizeof(float));
  int batch = 10000;
  Layer net[2];
  net[0] = relu(wrows, wcols, w1, b1);
  net[1] = soft(yrows, wrows, w2, b2);
  
  printf("Forward batch 10000:\n");
  forwtest(net, x, ytest, 2, xcols, 10000);
  // print3("test y", ytest);
  maxdiff("y", ysize, ytest, yorig);
  printf("Forward batch 1000:\n");
  forwtest(net, x, ytest, 2, xcols, 1000);
  maxdiff("y", ysize, ytest, yorig);
  printf("Forward batch 100:\n");
  forwtest(net, x, ytest, 2, xcols, 100);
  maxdiff("y", ysize, ytest, yorig);

  printf("Forwback batch 10000:\n");
  float *dw1, *dw2, *db1, *db2;
  file_id = H5Fopen("forwback10k.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
  h5read(file_id, "/dw1", &dw1, dims);
  h5read(file_id, "/dw2", &dw2, dims);
  h5read(file_id, "/db1", &db1, dims);
  h5read(file_id, "/db2", &db2, dims);
  // print3("orig dw1", dw1);
  // print3("orig db1", db1); 
  // print3("orig dw2", dw2);
  // print3("orig db2", db2); 

  int fbtest = 10000;
  forwbacktest(net, x, ygold, 2, fbtest, fbtest);

  float *gw1, *gw2, *gb1, *gb2;
  gw1 = gather(net[0]->dw, wrows*wcols);
  gw2 = gather(net[1]->dw, yrows*wrows);
  gb1 = gather(net[0]->db, wrows);
  gb2 = gather(net[1]->db, yrows);
  // print3("test dw1", gw1);
  // print3("test db1", gb1);
  // print3("test dw2", gw2);
  // print3("test db2", gb2);
  maxdiff("dw1", wrows*wcols, dw1, gw1);
  maxdiff("db1", wrows, db1, gb1);
  maxdiff("dw2", yrows*wrows, dw2, gw2);
  maxdiff("db2", yrows, db2, gb2);

  printf("Timing tests:\n");
  forwbacktest(net, x, ygold, 2, fbtest, fbtest);
  forwbacktest(net, x, ygold, 2, fbtest, fbtest);
  forwbacktest(net, x, ygold, 2, fbtest, fbtest);

  lfree(net[0]);
  lfree(net[1]);
}

