__global__ void _add_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void add_32_01(int n, float xi, float *y, float *z) {
    _add_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _add_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void add_64_01(int n, double xi, double *y, double *z) {
    _add_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _sub_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sub_32_01(int n, float xi, float *y, float *z) {
    _sub_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _sub_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sub_64_01(int n, double xi, double *y, double *z) {
    _sub_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _mul_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void mul_32_01(int n, float xi, float *y, float *z) {
    _mul_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _mul_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void mul_64_01(int n, double xi, double *y, double *z) {
    _mul_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _div_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void div_32_01(int n, float xi, float *y, float *z) {
    _div_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _div_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void div_64_01(int n, double xi, double *y, double *z) {
    _div_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _pow_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void pow_32_01(int n, float xi, float *y, float *z) {
    _pow_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _pow_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void pow_64_01(int n, double xi, double *y, double *z) {
    _pow_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _max_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void max_32_01(int n, float xi, float *y, float *z) {
    _max_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _max_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void max_64_01(int n, double xi, double *y, double *z) {
    _max_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _min_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void min_32_01(int n, float xi, float *y, float *z) {
    _min_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _min_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void min_64_01(int n, double xi, double *y, double *z) {
    _min_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _eq_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void eq_32_01(int n, float xi, float *y, float *z) {
    _eq_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _eq_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void eq_64_01(int n, double xi, double *y, double *z) {
    _eq_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _ne_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ne_32_01(int n, float xi, float *y, float *z) {
    _ne_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _ne_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ne_64_01(int n, double xi, double *y, double *z) {
    _ne_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _gt_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void gt_32_01(int n, float xi, float *y, float *z) {
    _gt_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _gt_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void gt_64_01(int n, double xi, double *y, double *z) {
    _gt_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _ge_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ge_32_01(int n, float xi, float *y, float *z) {
    _ge_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _ge_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ge_64_01(int n, double xi, double *y, double *z) {
    _ge_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _lt_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void lt_32_01(int n, float xi, float *y, float *z) {
    _lt_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _lt_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void lt_64_01(int n, double xi, double *y, double *z) {
    _lt_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _le_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void le_32_01(int n, float xi, float *y, float *z) {
    _le_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _le_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void le_64_01(int n, double xi, double *y, double *z) {
    _le_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _invxback_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void invxback_32_01(int n, float xi, float *y, float *z) {
    _invxback_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _invxback_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void invxback_64_01(int n, double xi, double *y, double *z) {
    _invxback_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _reluback_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void reluback_32_01(int n, float xi, float *y, float *z) {
    _reluback_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _reluback_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void reluback_64_01(int n, double xi, double *y, double *z) {
    _reluback_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _sigmback_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sigmback_32_01(int n, float xi, float *y, float *z) {
    _sigmback_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _sigmback_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sigmback_64_01(int n, double xi, double *y, double *z) {
    _sigmback_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _tanhback_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void tanhback_32_01(int n, float xi, float *y, float *z) {
    _tanhback_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _tanhback_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void tanhback_64_01(int n, double xi, double *y, double *z) {
    _tanhback_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _rpow_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void rpow_32_01(int n, float xi, float *y, float *z) {
    _rpow_32_01<<<256,256>>>(n,xi,y,z);
  }    
}
__global__ void _rpow_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void rpow_64_01(int n, double xi, double *y, double *z) {
    _rpow_64_01<<<256,256>>>(n,xi,y,z);
  }    
}
