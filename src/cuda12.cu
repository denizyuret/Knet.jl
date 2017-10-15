__global__ void _add_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void add_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _add_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _add_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void add_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _add_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _sub_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sub_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _sub_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _sub_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sub_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _sub_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _mul_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void mul_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _mul_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _mul_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void mul_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _mul_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _div_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void div_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _div_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _div_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void div_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _div_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _pow_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void pow_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _pow_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _pow_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void pow_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _pow_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _max_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void max_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _max_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _max_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void max_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _max_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _min_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void min_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _min_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _min_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void min_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _min_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _eq_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void eq_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _eq_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _eq_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void eq_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _eq_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _ne_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ne_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _ne_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _ne_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ne_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _ne_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _gt_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void gt_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _gt_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _gt_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void gt_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _gt_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _ge_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ge_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _ge_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _ge_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ge_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _ge_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _lt_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void lt_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _lt_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _lt_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void lt_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _lt_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _le_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void le_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _le_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _le_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void le_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _le_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _invxback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void invxback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _invxback_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _invxback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void invxback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _invxback_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _reluback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void reluback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _reluback_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _reluback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void reluback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _reluback_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _sigmback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sigmback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _sigmback_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _sigmback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sigmback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _sigmback_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _tanhback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void tanhback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _tanhback_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _tanhback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void tanhback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _tanhback_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _rpow_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void rpow_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _rpow_32_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
__global__ void _rpow_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void rpow_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _rpow_64_12<<<256,256>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
