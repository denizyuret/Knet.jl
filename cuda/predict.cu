#include <stdio.h>
#include "jnet.h"
#include "jnet_h5.h"
#define BATCH 10000

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: predict x layer1 layer2 ... y  where each arg is an hdf5 file and y will be overwritten\n");
    exit(0);
  }
  float *x; int xrows, xcols;
  h5read(argv[1], &xrows, &xcols, &x);
  int nlayers = argc - 3;
  Layer *net = (Layer *) calloc(nlayers, sizeof(Layer));
  for (int l = 0; l < nlayers; l++) net[l] = h5read_layer(argv[l+2]);
  int yrows = net[nlayers-1]->wrows;
  float *y = (float *) malloc(yrows * xcols * sizeof(float));
  forward(net, x, y, nlayers, xcols, BATCH);
  h5write(argv[argc-1], yrows, xcols, y);
  for (int l = 0; l < nlayers; l++) lfree(net[l]);
  free(x); free(y); free(net);
}
