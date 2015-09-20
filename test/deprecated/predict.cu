#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "kunet.h"
#include "kunet_h5.h"

static clock_t t0;
#define tic (t0 = clock())
#define toc fprintf(stderr, "%g seconds\n", (double)(clock()-t0)/CLOCKS_PER_SEC)

const char *usage =
  "Usage: %s [-b batchsize] x layer1 layer2 ... y\n"  
  "where each of x layer1 ... y is an hdf5 file\n"
  "and y will be overwritten.";

int main(int argc, char **argv) {
  int batch = 0;
  int opt;
  while((opt = getopt(argc, argv, "b:")) != -1) {
    switch(opt) {
    case 'b': batch = atoi(optarg); break;
    default: fprintf(stderr, usage, argv[0]); exit(EXIT_FAILURE);
    }
  }
  if (argc - optind < 3) {
    fprintf(stderr, usage, argv[0]); exit(EXIT_FAILURE);
  }

  float *x; int xrows, xcols;
  fprintf(stderr, "Reading %s... ", argv[optind]);
  tic; h5read(argv[optind++], &xrows, &xcols, &x); toc;
  
  int nlayers = argc - optind - 1;
  Layer *net = (Layer *) calloc(nlayers, sizeof(Layer));
  fprintf(stderr, "Reading "); tic;
  for (int l = 0; l < nlayers; l++) {
    fprintf(stderr, "%s... ", argv[optind + l]);
    net[l] = h5read_layer(argv[optind + l]);
  }
  toc;

  int yrows = net[nlayers-1]->wrows;
  float *y = (float *) malloc(yrows * xcols * sizeof(float));
  fprintf(stderr, "Predicting... "); 
  tic; forward(net, x, y, nlayers, xcols, (batch ? batch : xcols)); toc;

  fprintf(stderr, "Saving %s... ", argv[argc-1]); t0 = clock();
  tic; h5write(argv[argc-1], yrows, xcols, y); toc;
  for (int l = 0; l < nlayers; l++) lfree(net[l]);
  free(x); free(y); free(net);
}
