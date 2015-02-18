#ifndef JNET_H5_H_
#define JNET_H5_H_

#include "kunet.h"
Layer h5read_layer(const char *fname);
void h5write_layer(const char *fname, Layer l);
void h5read(const char *fname, int *nrows, int *ncols, float **data);
void h5write(const char *fname, int nrows, int ncols, float *data);

#endif
