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

#ifdef __cplusplus
}
#endif
