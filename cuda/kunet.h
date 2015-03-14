#ifndef JNET_H_
#define JNET_H_

#define DEFAULT_LEARNING_RATE 0.01
#define ADAGRAD_EPSILON 1e-8

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { NOYF=0, RELU=1, SOFT=2, SIGM=3 } Yfunc;
typedef enum { NOXF=0, DROP=1 } Xfunc;

typedef struct LayerS {
  int wrows, wcols;	// size of w matrix
  int xcols, acols;	// actual and allocated x columns

  Xfunc xfunc;		// type of preprocessing function, e.g. dropout
  Yfunc yfunc;		// type of activation function, e.g. relu
  float *w;		// weight matrix (wrows,wcols)
  float *b;		// bias vector (wrows)
  float *x;		// last input (wcols,xcols)
  float *y;		// last output (wrows,xcols)
  float *xones;		// vector of ones for bias calculation (xcols)
  float *xmask;		// input mask for dropout

  float *dw;		// gradient wrt weight matrix
  float *db;		// gradient wrt bias vector
  float *dx;		// gradient wrt input
  float *dy;		// gradient wrt output

  float *dw1;		// moving average of gradients for momentum
  float *dw2;		// sum of squared gradients for adagrad
  float *db1;		// moving average of gradients for momentum
  float *db2;		// sum of squared gradients for adagrad

  float adagrad;	// [0] apply adagrad if nonzero, using value as epsilon
  float nesterov;	// [0] nesterov if nonzero, using value as momentum
  float learningRate;	// [0.01]
  float momentum;	// [0]
  float dropout;	// [0] probability of dropping inputs
  float maxnorm;	// [0] default=0, acts like inf
  float L1, L2;		// [0,0] L1,L2 regularization
} *Layer;

Layer layer(Xfunc xfunc, Yfunc yfunc, int wrows, int wcols, float *w, float *b);
Layer relu(int wrows, int wcols, float *w, float *b);
Layer soft(int wrows, int wcols, float *w, float *b);
void lfree(Layer l);
void lclean(Layer l);
int lsize(Layer l, int i);
float *lforw(Layer l, float *x, int xcols);
float *lback(Layer l, float *dy, int return_dx);
float *ldrop(Layer l, float *x, int xcols);
void lupdate(Layer l);
void forward(Layer *net, float *x, float *y, int nlayer, int xcols, int batch);
void forwback(Layer *net, float *x, float *y, int nlayer, int xcols, int batch);
void train(Layer *net, float *x, float *y, int nlayer, int xcols, int batch);

void reluforw(int n, float *y);
void reluback(int n, float *y, float *dy);
void logpforw(int nrows, int ncols, float *y);
void softback(int nrows, int ncols, float *y, float *dy);
void logploss(int nrows, int ncols, float *y, float *dy);
void l1reg(int n, float l1, float *w, float *dw);
void adagrad(int n, float eps, float *dw2, float *dw);
void fill(int n, float val, float *x);
void add1(int n, float val, float *x);
void drop(int n, float *x, float *xmask, float dropout, float scale);
void badd(int nrows, int ncols, float *y, float *b);
void bsum(int nrows, int ncols, float *y, float *b);
void randfill(int n, float *x);
void gpuseed(unsigned long long seed);

void set_adagrad(Layer l, float a);
void set_nesterov(Layer l, float n);
void set_learningRate(Layer l, float lr);
void set_momentum(Layer l, float m);
void set_dropout(Layer l, float d);
void set_maxnorm(Layer l, float m);
void set_L1(Layer l, float m);
void set_L2(Layer l, float m);

#ifdef __cplusplus
}
#endif

#endif
