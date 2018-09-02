#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include "kunet.h"
#include "kunet_h5.h"

static clock_t t0;
#define tic (t0 = clock())
#define toc fprintf(stderr, "%g seconds\n", (double)(clock()-t0)/CLOCKS_PER_SEC)

const char *usage = "Usage: %s [opts] x layer1 layer2 ... y\n";

int main(int argc, char **argv) {
  Layer o = (Layer) calloc(1, sizeof(struct LayerS));
  int batch = 0;
  int iters = 0;
  const char *output = "train.out";
  
  int opt;
  while((opt = getopt(argc, argv, "o:b:i:m:d:x:l:1:2:a:n:")) != -1) {
    switch(opt) {
    case 'b': batch = atoi(optarg); break;
    case 'i': iters = atoi(optarg); break;
    case 'o': output = optarg; break;
    case 'l': o->learningRate = atof(optarg); break;
    case 'a': o->adagrad = atof(optarg); break;
    case 'n': o->nesterov = atof(optarg); break;
    case 'm': o->momentum = atof(optarg); break;
    case 'd': o->dropout = atof(optarg); break;
    case 'x': o->maxnorm = atof(optarg); break;
    case '1': o->L1 = atof(optarg); break;
    case '2': o->L2 = atof(optarg); break;
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
    if (o->adagrad) net[l]->adagrad = o->adagrad;
    if (o->nesterov) net[l]->nesterov = o->nesterov;
    if (o->learningRate) net[l]->learningRate = o->learningRate;
    if (o->momentum) net[l]->momentum = o->momentum;
    if (o->maxnorm) net[l]->maxnorm = o->maxnorm;
    if (o->L1) net[l]->L1 = o->L1;
    if (o->L2) net[l]->L2 = o->L2;
    if (o->dropout) {
      net[l]->dropout = o->dropout;
      net[l]->xfunc = DROP;
    }
  }
  optind += nlayers;
  toc;
  // if (o->dropout) set_seed(1);

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

/* Debugging */

void printn(int n, float *gptr) {
  float *cptr = (float*) malloc(n * sizeof(float));
  cudaMemcpy(cptr, gptr, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) printf("%g ", cptr[i]);
  putchar('\n');
  free(cptr);
}
