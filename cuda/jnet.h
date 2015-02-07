#ifdef __cplusplus
extern "C" {
#endif

typedef struct LayerS *Layer;

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
