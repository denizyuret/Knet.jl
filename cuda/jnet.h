#ifdef __cplusplus
extern "C" {
#endif

typedef enum { NOOP=0, RELU=1, SOFT=2 } LayerType;

typedef struct LayerS {
  int wrows, wcols, xcols; // size params

  LayerType type;	// type of activation function	
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

  int adagrad;		// [0] adagrad during weight updates
  int nesterov;		// [0] nesterov during weight updates
  float learningRate;	// [0.01]
  float momentum;	// [0]
  float dropout;	// [0] probability of dropping inputs
  float maxnorm;	// [0] default=0, acts like inf
  float L1, L2;		// [0,0] L1,L2 regularization
} *Layer;

Layer layer(LayerType type, int wrows, int wcols, float *w, float *b);
Layer relu(int wrows, int wcols, float *w, float *b);
Layer soft(int wrows, int wcols, float *w, float *b);
void lfree(Layer l);
int lsize(Layer l, int i);
void forward(Layer *net, float *x, float *y, int nlayer, int xcols, int batch);
void forwback(Layer *net, float *x, float *y, int nlayer, int xcols, int batch);
void update(Layer l);
float *lforw(Layer l, float *x, int xcols);
float *lback(Layer l, float *dy, int dx);

void set_adagrad(Layer l, int i);
void set_nesterov(Layer l, int i);
void set_learningRate(Layer l, float lr);
void set_momentum(Layer l, float m);
void set_dropout(Layer l, float d);
void set_maxnorm(Layer l, float m);
void set_L1(Layer l, float m);
void set_L2(Layer l, float m);

#ifdef __cplusplus
}
#endif
