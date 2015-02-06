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
float *forw(Layer l, float *x, int xcols);
float *back(Layer l, float *dy, int dx);

void adagrad(Layer l, int i);
void nesterov(Layer l, int i);
void learningRate(Layer l, float lr);
void momentum(Layer l, float m);
void dropout(Layer l, float d);
void maxnorm(Layer l, float m);
void L1(Layer l, float m);
void L2(Layer l, float m);

#ifdef __cplusplus
}
#endif
