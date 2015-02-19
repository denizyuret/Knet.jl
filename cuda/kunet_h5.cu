/* Read and write layers and arrays in HDF5 format */
/* TODO: add error handling */
/* TODO: do not hardcode float */

#include <assert.h>
#include <cuda_runtime.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "kunet.h"
#include "kunet_h5.h"

static inline void *copy_to_gpu(void *cptr, size_t n) {
  if (cptr == NULL || n == 0) return NULL;
  void *gptr; cudaMalloc((void **) &gptr, n);
  cudaMemcpy(gptr, cptr, n, cudaMemcpyHostToDevice);
  return gptr;
}

static inline void *copy_to_cpu(void *gptr, size_t n) {
  if (gptr == NULL || n == 0) return NULL;
  void *cptr = malloc(n);
  cudaMemcpy(cptr, gptr, n, cudaMemcpyDeviceToHost);
  return cptr;
}

static inline void check_dims(int *nptr, int n) {
  if (nptr == NULL) assert(n == 1);
  else if (*nptr == 0) *nptr = n;
  else assert(*nptr == n);
}

static inline void h5read_to_gpu(hid_t id, const char *name, int *nrows, int *ncols, float **data) {
  if (H5LTfind_dataset(id, (name+1))) {
    hsize_t dims[2];
    H5LTget_dataset_info(id,name,dims,NULL,NULL);
    check_dims(nrows, dims[1]);
    check_dims(ncols, dims[0]);
    int size = dims[0]*dims[1]*sizeof(float);
    if (size > 0) {
      float *cpuArray = (float *) malloc(size);
      H5LTread_dataset_float(id, name, cpuArray);
      *data = (float *) copy_to_gpu(cpuArray, size);
      free(cpuArray);
    } else {
      *data = NULL;
    }
  } else {
    *data = NULL;
  }
}

// TODO: these will break if attr is not of the right type and size=1

static inline void h5read_int(hid_t id, const char *attr, int *ptr) {
  if (H5LTfind_attribute(id, attr))
    H5LTget_attribute_int(id, "/", attr, ptr);
}

static inline void h5read_float(hid_t id, const char *attr, float *ptr) {
  if (H5LTfind_attribute(id, attr))
    H5LTget_attribute_float(id, "/", attr, ptr);
}

Layer h5read_layer(const char *fname) {
  Layer l = layer(NOXF, NOYF, 0, 0, NULL, NULL);
  hid_t id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  h5read_int(id, "xfunc", (int*)&l->xfunc);
  h5read_int(id, "yfunc", (int*)&l->yfunc);
  h5read_float(id, "nesterov", &l->nesterov);
  h5read_float(id, "adagrad", &l->adagrad);
  h5read_float(id, "learningRate", &l->learningRate);
  h5read_float(id, "momentum", &l->momentum);
  h5read_float(id, "dropout", &l->dropout);
  h5read_float(id, "maxnorm", &l->maxnorm);
  h5read_float(id, "L1", &l->L1);
  h5read_float(id, "L2", &l->L2);
  h5read_to_gpu(id, "/w", &l->wrows, &l->wcols, &l->w);
  h5read_to_gpu(id, "/b", &l->wrows, NULL, &l->b);
  h5read_to_gpu(id, "/dw", &l->wrows, &l->wcols, &l->dw);
  h5read_to_gpu(id, "/db", &l->wrows, NULL, &l->db);
  h5read_to_gpu(id, "/dw1", &l->wrows, &l->wcols, &l->dw1);
  h5read_to_gpu(id, "/db1", &l->wrows, NULL, &l->db1);
  h5read_to_gpu(id, "/dw2", &l->wrows, &l->wcols, &l->dw2);
  h5read_to_gpu(id, "/db2", &l->wrows, NULL, &l->db2);
  H5Fclose(id);
  return l;
}

static inline void h5write_from_gpu(hid_t id, const char *name, int nrows, int ncols, float *data) {
  if (data == NULL) return;
  float *cptr = (float *) copy_to_cpu(data, nrows * ncols * sizeof(float));
  hsize_t dims[2] = { ncols, nrows };
  H5LTmake_dataset_float(id, name, 2, dims, cptr);
  free(cptr);
}

static inline void h5write_int(hid_t id, const char *name, int val, int defval) {
  if (val != defval) H5LTset_attribute_int(id, "/", name, &val, 1);
}

static inline void h5write_float(hid_t id, const char *name, float val, float defval) {
  if (val != defval) H5LTset_attribute_float(id, "/", name, &val, 1);
}

void h5write_layer(const char *fname, Layer l) {
  hid_t id = H5Fcreate (fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  int xfunc = (int) l->xfunc;
  int yfunc = (int) l->yfunc;
  h5write_int(id, "xfunc", xfunc, 0);
  h5write_int(id, "yfunc", yfunc, 0);
  h5write_float(id, "nesterov", l->nesterov, 0);
  h5write_float(id, "adagrad", l->adagrad, 0);
  h5write_float(id, "learningRate", l->learningRate, DEFAULT_LEARNING_RATE);
  h5write_float(id, "momentum", l->momentum, 0);
  h5write_float(id, "dropout", l->dropout, 0);
  h5write_float(id, "maxnorm", l->maxnorm, 0);
  h5write_float(id, "L1", l->L1, 0);
  h5write_float(id, "L2", l->L2, 0);
  h5write_from_gpu(id, "/w", l->wrows, l->wcols, l->w);
  h5write_from_gpu(id, "/b", l->wrows, 1, l->b);
  h5write_from_gpu(id, "/dw", l->wrows, l->wcols, l->dw);
  h5write_from_gpu(id, "/db", l->wrows, 1, l->db);
  h5write_from_gpu(id, "/dw1", l->wrows, l->wcols, l->dw1);
  h5write_from_gpu(id, "/db1", l->wrows, 1, l->db1);
  h5write_from_gpu(id, "/dw2", l->wrows, l->wcols, l->dw2);
  h5write_from_gpu(id, "/db2", l->wrows, 1, l->db2);
  H5Fclose(id);
}

void h5read(const char *fname, int *nrows, int *ncols, float **data) {
  const char *name = "/data";
  hid_t id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hsize_t dims[2]; H5LTget_dataset_info(id,name,dims,NULL,NULL);
  int size = dims[0]*dims[1]*sizeof(float);
  if (size > 0) {
    *nrows = dims[1];
    *ncols = dims[0];
    *data = (float *) malloc(size);
    H5LTread_dataset_float(id, name, *data);
  } else {
    *nrows = 0;
    *ncols = 0;
    *data = NULL;
  }
  H5Fclose(id);
}

void h5write(const char *fname, int nrows, int ncols, float *data) {
  const char *name = "/data";
  hid_t id = H5Fcreate (fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t dims[2] = { ncols, nrows };
  H5LTmake_dataset_float(id, name, 2, dims, data);
  H5Fclose(id);
}
