#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include "jnet.h"
#include "jnet_h5.h"

static clock_t t0;
#define tic (t0 = clock())
#define toc fprintf(stderr, "%g seconds\n", (double)(clock()-t0)/CLOCKS_PER_SEC)

const char *usage =
  "Usage: %s [opts] x layer1 layer2 ... y\n"  
  "where each of x layer1 ... y is an hdf5 file\n"
  "-b batchsize (default: size of dataset)\n"
  "-o prefix (default: train.out)\n"
  "   the new layers will be saved in prefix1.h5, prefix2.h5, ... etc.\n"
  "-a (adagrad, default:false)\n"
  "-i iters (default: until one epoch is completed)\n";

int main(int argc, char **argv) {
  int batch = 0;
  int adagrad = 0;
  int iters = 0;
  const char *output = "train.out";
  int opt;
  while((opt = getopt(argc, argv, "o:b:i:a")) != -1) {
    switch(opt) {
    case 'b': batch = atoi(optarg); break;
    case 'i': iters = atoi(optarg); break;
    case 'o': output = optarg; break;
    case 'a': adagrad = 1; break;
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
    if (adagrad) net[l]->adagrad = 1;
  }
  optind += nlayers;
  toc;

  float *y; int yrows, ycols;
  fprintf(stderr, "Reading %s... ", argv[optind]);
  tic; h5read(argv[optind++], &yrows, &ycols, &y); toc;
  assert(yrows == net[nlayers-1]->wrows);
  assert(ycols == xcols);

  if (batch == 0) batch = xcols;
  if (iters > 0) { 
    assert(xcols >= batch * iters); 
    fprintf(stderr, "Training %d iters of %d batches... ", iters, batch);
    xcols = batch * iters; 
  } else {
    fprintf(stderr, "Training a single epoch with batch=%d... ", batch);
  }
  tic; train(net, x, y, nlayers, xcols, batch); toc;

  char *fname = (char *) malloc(strlen(output) + 128);
  fprintf(stderr, "Saving resulting layers "); tic;
  for (int l = 0; l < nlayers; l++) {
    sprintf(fname, "%s%d.h5", output, l+1);
    fprintf(stderr, "%s... ", fname);
    h5write_layer(fname, net[l]);
  }
  toc;
  free(fname);

  for (int l = 0; l < nlayers; l++) lfree(net[l]);
  free(x); free(y); free(net);
}
