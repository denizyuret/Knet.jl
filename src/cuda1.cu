__global__ void _abs2_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi*xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void abs2_32(int n, float *x, float *y) {
    if (n>0) _abs2_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _abs2_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi*xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void abs2_64(int n, double *x, double *y) {
    if (n>0) _abs2_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _abs_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi<0?-xi:xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void abs_32(int n, float *x, float *y) {
    if (n>0) _abs_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _abs_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi<0?-xi:xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void abs_64(int n, double *x, double *y) {
    if (n>0) _abs_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _acos_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = acos(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void acos_32(int n, float *x, float *y) {
    if (n>0) _acos_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _acos_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = acos(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void acos_64(int n, double *x, double *y) {
    if (n>0) _acos_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _acosh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = acosh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void acosh_32(int n, float *x, float *y) {
    if (n>0) _acosh_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _acosh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = acosh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void acosh_64(int n, double *x, double *y) {
    if (n>0) _acosh_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _asin_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = asin(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void asin_32(int n, float *x, float *y) {
    if (n>0) _asin_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _asin_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = asin(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void asin_64(int n, double *x, double *y) {
    if (n>0) _asin_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _asinh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = asinh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void asinh_32(int n, float *x, float *y) {
    if (n>0) _asinh_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _asinh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = asinh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void asinh_64(int n, double *x, double *y) {
    if (n>0) _asinh_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _atan_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = atan(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void atan_32(int n, float *x, float *y) {
    if (n>0) _atan_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _atan_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = atan(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void atan_64(int n, double *x, double *y) {
    if (n>0) _atan_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _atanh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = atanh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void atanh_32(int n, float *x, float *y) {
    if (n>0) _atanh_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _atanh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = atanh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void atanh_64(int n, double *x, double *y) {
    if (n>0) _atanh_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _cbrt_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = cbrt(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void cbrt_32(int n, float *x, float *y) {
    if (n>0) _cbrt_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _cbrt_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = cbrt(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void cbrt_64(int n, double *x, double *y) {
    if (n>0) _cbrt_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _ceil_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = ceil(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ceil_32(int n, float *x, float *y) {
    if (n>0) _ceil_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _ceil_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = ceil(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ceil_64(int n, double *x, double *y) {
    if (n>0) _ceil_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _cos_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = cos(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void cos_32(int n, float *x, float *y) {
    if (n>0) _cos_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _cos_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = cos(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void cos_64(int n, double *x, double *y) {
    if (n>0) _cos_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _cosh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = cosh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void cosh_32(int n, float *x, float *y) {
    if (n>0) _cosh_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _cosh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = cosh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void cosh_64(int n, double *x, double *y) {
    if (n>0) _cosh_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _cospi_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = cospi(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void cospi_32(int n, float *x, float *y) {
    if (n>0) _cospi_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _cospi_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = cospi(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void cospi_64(int n, double *x, double *y) {
    if (n>0) _cospi_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _exp_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = exp(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void exp_32(int n, float *x, float *y) {
    if (n>0) _exp_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _exp_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = exp(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void exp_64(int n, double *x, double *y) {
    if (n>0) _exp_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _exp10_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = exp10(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void exp10_32(int n, float *x, float *y) {
    if (n>0) _exp10_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _exp10_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = exp10(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void exp10_64(int n, double *x, double *y) {
    if (n>0) _exp10_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _exp2_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = exp2(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void exp2_32(int n, float *x, float *y) {
    if (n>0) _exp2_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _exp2_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = exp2(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void exp2_64(int n, double *x, double *y) {
    if (n>0) _exp2_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _expm1_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = expm1(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void expm1_32(int n, float *x, float *y) {
    if (n>0) _expm1_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _expm1_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = expm1(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void expm1_64(int n, double *x, double *y) {
    if (n>0) _expm1_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _floor_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = floor(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void floor_32(int n, float *x, float *y) {
    if (n>0) _floor_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _floor_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = floor(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void floor_64(int n, double *x, double *y) {
    if (n>0) _floor_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _invx_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = 1/xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void invx_32(int n, float *x, float *y) {
    if (n>0) _invx_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _invx_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = 1/xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void invx_64(int n, double *x, double *y) {
    if (n>0) _invx_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _log_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = log(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void log_32(int n, float *x, float *y) {
    if (n>0) _log_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _log_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = log(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void log_64(int n, double *x, double *y) {
    if (n>0) _log_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _log10_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = log10(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void log10_32(int n, float *x, float *y) {
    if (n>0) _log10_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _log10_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = log10(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void log10_64(int n, double *x, double *y) {
    if (n>0) _log10_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _log1p_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = log1p(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void log1p_32(int n, float *x, float *y) {
    if (n>0) _log1p_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _log1p_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = log1p(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void log1p_64(int n, double *x, double *y) {
    if (n>0) _log1p_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _log2_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = log2(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void log2_32(int n, float *x, float *y) {
    if (n>0) _log2_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _log2_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = log2(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void log2_64(int n, double *x, double *y) {
    if (n>0) _log2_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _neg_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = -xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void neg_32(int n, float *x, float *y) {
    if (n>0) _neg_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _neg_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = -xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void neg_64(int n, double *x, double *y) {
    if (n>0) _neg_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _relu_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void relu_32(int n, float *x, float *y) {
    if (n>0) _relu_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _relu_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void relu_64(int n, double *x, double *y) {
    if (n>0) _relu_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _round_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = round(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void round_32(int n, float *x, float *y) {
    if (n>0) _round_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _round_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = round(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void round_64(int n, double *x, double *y) {
    if (n>0) _round_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sigm_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi>=0?1/(1+exp(-xi)):(exp(xi)/(1+exp(xi))));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sigm_32(int n, float *x, float *y) {
    if (n>0) _sigm_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sigm_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi>=0?1/(1+exp(-xi)):(exp(xi)/(1+exp(xi))));
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sigm_64(int n, double *x, double *y) {
    if (n>0) _sigm_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sign_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi>0?1:xi<0?-1:0);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sign_32(int n, float *x, float *y) {
    if (n>0) _sign_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sign_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi>0?1:xi<0?-1:0);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sign_64(int n, double *x, double *y) {
    if (n>0) _sign_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sin_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = sin(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sin_32(int n, float *x, float *y) {
    if (n>0) _sin_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sin_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = sin(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sin_64(int n, double *x, double *y) {
    if (n>0) _sin_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sinh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = sinh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sinh_32(int n, float *x, float *y) {
    if (n>0) _sinh_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sinh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = sinh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sinh_64(int n, double *x, double *y) {
    if (n>0) _sinh_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sinpi_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = sinpi(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sinpi_32(int n, float *x, float *y) {
    if (n>0) _sinpi_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sinpi_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = sinpi(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sinpi_64(int n, double *x, double *y) {
    if (n>0) _sinpi_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sqrt_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = sqrt(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sqrt_32(int n, float *x, float *y) {
    if (n>0) _sqrt_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _sqrt_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = sqrt(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sqrt_64(int n, double *x, double *y) {
    if (n>0) _sqrt_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _tan_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = tan(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void tan_32(int n, float *x, float *y) {
    if (n>0) _tan_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _tan_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = tan(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void tan_64(int n, double *x, double *y) {
    if (n>0) _tan_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _tanh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = tanh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void tanh_32(int n, float *x, float *y) {
    if (n>0) _tanh_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _tanh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = tanh(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void tanh_64(int n, double *x, double *y) {
    if (n>0) _tanh_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _trunc_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = trunc(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void trunc_32(int n, float *x, float *y) {
    if (n>0) _trunc_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _trunc_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = trunc(xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void trunc_64(int n, double *x, double *y) {
    if (n>0) _trunc_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _fill_32(int n, float x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = x;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void fill_32(int n, float x, float *y) {
    if (n>0) _fill_32<<<256,256>>>(n,x,y);
  }    
}
__global__ void _fill_64(int n, double x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = x;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void fill_64(int n, double x, double *y) {
    if (n>0) _fill_64<<<256,256>>>(n,x,y);
  }    
}
__global__ void _xfill_32(int nrows, int ncols, float x, float *y, int incy) {
  int row, col, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    yidx = row + col * incy;
    y[yidx] = x;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void xfill_32(int nrows, int ncols, float x, float *y, int incy) {
    if (nrows>0 && ncols>0) _xfill_32<<<256,256>>>(nrows, ncols, x, y, incy);
  }    
}
__global__ void _xfill_64(int nrows, int ncols, double x, double *y, int incy) {
  int row, col, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    yidx = row + col * incy;
    y[yidx] = x;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void xfill_64(int nrows, int ncols, double x, double *y, int incy) {
    if (nrows>0 && ncols>0) _xfill_64<<<256,256>>>(nrows, ncols, x, y, incy);
  }    
}
__global__ void _xcopy(int nrows, int ncols, const char *x, int incx, char *y, int incy) {
  int row, col, xidx, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    xidx = row + col * incx;
    yidx = row + col * incy;
    y[yidx] = x[xidx];
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void xcopy(int nrows, int ncols, const void *x, int incx, void *y, int incy) {
    if (nrows>0 && ncols>0) _xcopy<<<256,256>>>(nrows,ncols,(char*)x,incx,(char*)y,incy);
  }    
}
__global__ void _permutedims_2D_1_2_32(float* x, int dimx1, int dimx2, float* y, int dimy1) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2; v += blockDim.x * gridDim.x) {

    //From 1D to 2D indices
    int i = v % dimx1;
    int j = (v-i) / dimx1;

    //Calculate destination
    int destIndex = i + j*dimy1;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_2D_1_2_32(float* x, int dimx1, int dimx2, float* y, int dimy1) {
    _permutedims_2D_1_2_32<<<256,256>>>(x,dimx1,dimx2,y,dimy1);
  }    
}
__global__ void _permutedims_2D_1_2_64(double* x, int dimx1, int dimx2, double* y, int dimy1) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2; v += blockDim.x * gridDim.x) {

    //From 1D to 2D indices
    int i = v % dimx1;
    int j = (v-i) / dimx1;

    //Calculate destination
    int destIndex = i + j*dimy1;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_2D_1_2_64(double* x, int dimx1, int dimx2, double* y, int dimy1) {
    _permutedims_2D_1_2_64<<<256,256>>>(x,dimx1,dimx2,y,dimy1);
  }    
}
__global__ void _permutedims_2D_2_1_32(float* x, int dimx1, int dimx2, float* y, int dimy1) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2; v += blockDim.x * gridDim.x) {

    //From 1D to 2D indices
    int i = v % dimx1;
    int j = (v-i) / dimx1;

    //Calculate destination
    int destIndex = j + i*dimy1;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_2D_2_1_32(float* x, int dimx1, int dimx2, float* y, int dimy1) {
    _permutedims_2D_2_1_32<<<256,256>>>(x,dimx1,dimx2,y,dimy1);
  }    
}
__global__ void _permutedims_2D_2_1_64(double* x, int dimx1, int dimx2, double* y, int dimy1) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2; v += blockDim.x * gridDim.x) {

    //From 1D to 2D indices
    int i = v % dimx1;
    int j = (v-i) / dimx1;

    //Calculate destination
    int destIndex = j + i*dimy1;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_2D_2_1_64(double* x, int dimx1, int dimx2, double* y, int dimy1) {
    _permutedims_2D_2_1_64<<<256,256>>>(x,dimx1,dimx2,y,dimy1);
  }    
}
__global__ void _permutedims_3D_1_2_3_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = i + j*dimy1 + k*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_1_2_3_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
    _permutedims_3D_1_2_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_1_2_3_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = i + j*dimy1 + k*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_1_2_3_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
    _permutedims_3D_1_2_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_1_3_2_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = i + k*dimy1 + j*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_1_3_2_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
    _permutedims_3D_1_3_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_1_3_2_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = i + k*dimy1 + j*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_1_3_2_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
    _permutedims_3D_1_3_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_2_1_3_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = j + i*dimy1 + k*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_2_1_3_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
    _permutedims_3D_2_1_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_2_1_3_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = j + i*dimy1 + k*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_2_1_3_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
    _permutedims_3D_2_1_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_2_3_1_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = j + k*dimy1 + i*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_2_3_1_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
    _permutedims_3D_2_3_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_2_3_1_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = j + k*dimy1 + i*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_2_3_1_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
    _permutedims_3D_2_3_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_3_1_2_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = k + i*dimy1 + j*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_3_1_2_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
    _permutedims_3D_3_1_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_3_1_2_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = k + i*dimy1 + j*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_3_1_2_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
    _permutedims_3D_3_1_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_3_2_1_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = k + j*dimy1 + i*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_3_2_1_32(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2) {
    _permutedims_3D_3_2_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_3D_3_2_1_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = k + j*dimy1 + i*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_3D_3_2_1_64(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2) {
    _permutedims_3D_3_2_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
__global__ void _permutedims_4D_1_2_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + j*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_2_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_2_3_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_2_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + j*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_2_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_2_3_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_2_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + j*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_2_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_2_4_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_2_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + j*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_2_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_2_4_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_3_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + k*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_3_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_3_2_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_3_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + k*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_3_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_3_2_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_3_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + k*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_3_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_3_4_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_3_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + k*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_3_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_3_4_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_4_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + l*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_4_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_4_2_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_4_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + l*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_4_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_4_2_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_4_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + l*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_4_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_4_3_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_1_4_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = i + l*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_1_4_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_1_4_3_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_1_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + i*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_1_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_1_3_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_1_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + i*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_1_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_1_3_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_1_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + i*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_1_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_1_4_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_1_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + i*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_1_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_1_4_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_3_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + k*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_3_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_3_1_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_3_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + k*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_3_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_3_1_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_3_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + k*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_3_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_3_4_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_3_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + k*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_3_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_3_4_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_4_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + l*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_4_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_4_1_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_4_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + l*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_4_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_4_1_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_4_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + l*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_4_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_4_3_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_2_4_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = j + l*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_2_4_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_2_4_3_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_1_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + i*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_1_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_1_2_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_1_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + i*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_1_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_1_2_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_1_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + i*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_1_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_1_4_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_1_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + i*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_1_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_1_4_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_2_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + j*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_2_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_2_1_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_2_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + j*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_2_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_2_1_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_2_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + j*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_2_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_2_4_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_2_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + j*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_2_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_2_4_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_4_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + l*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_4_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_4_1_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_4_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + l*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_4_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_4_1_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_4_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + l*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_4_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_4_2_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_3_4_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = k + l*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_3_4_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_3_4_2_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_1_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + i*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_1_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_1_2_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_1_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + i*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_1_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_1_2_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_1_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + i*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_1_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_1_3_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_1_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + i*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_1_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_1_3_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_2_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + j*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_2_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_2_1_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_2_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + j*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_2_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_2_1_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_2_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + j*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_2_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_2_3_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_2_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + j*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_2_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_2_3_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_3_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + k*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_3_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_3_1_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_3_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + k*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_3_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_3_1_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_3_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + k*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_3_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_3_2_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_4D_4_3_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = l + k*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_4D_4_3_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims_4D_4_3_2_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
__global__ void _permutedims_5D_1_2_3_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_3_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_3_4_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_3_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_3_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_3_4_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_3_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_3_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_3_5_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_3_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_3_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_3_5_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_4_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_4_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_4_3_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_4_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_4_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_4_3_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_4_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_4_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_4_5_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_4_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_4_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_4_5_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_5_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_5_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_5_3_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_5_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_5_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_5_3_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_5_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_5_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_5_4_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_2_5_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + j*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_2_5_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_2_5_4_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_2_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_2_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_2_4_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_2_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_2_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_2_4_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_2_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_2_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_2_5_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_2_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_2_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_2_5_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_4_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_4_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_4_2_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_4_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_4_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_4_2_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_4_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_4_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_4_5_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_4_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_4_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_4_5_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_5_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_5_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_5_2_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_5_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_5_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_5_2_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_5_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_5_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_5_4_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_3_5_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + k*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_3_5_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_3_5_4_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_2_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_2_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_2_3_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_2_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_2_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_2_3_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_2_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_2_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_2_5_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_2_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_2_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_2_5_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_3_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_3_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_3_2_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_3_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_3_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_3_2_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_3_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_3_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_3_5_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_3_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_3_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_3_5_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_5_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_5_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_5_2_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_5_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_5_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_5_2_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_5_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_5_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_5_3_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_4_5_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + l*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_4_5_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_4_5_3_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_2_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_2_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_2_3_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_2_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_2_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_2_3_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_2_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_2_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_2_4_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_2_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_2_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_2_4_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_3_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_3_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_3_2_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_3_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_3_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_3_2_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_3_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_3_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_3_4_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_3_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_3_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_3_4_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_4_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_4_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_4_2_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_4_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_4_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_4_2_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_4_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_4_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_4_3_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_1_5_4_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = i + m*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_1_5_4_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_1_5_4_3_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_3_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_3_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_3_4_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_3_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_3_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_3_4_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_3_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_3_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_3_5_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_3_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_3_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_3_5_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_4_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_4_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_4_3_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_4_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_4_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_4_3_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_4_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_4_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_4_5_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_4_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_4_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_4_5_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_5_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_5_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_5_3_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_5_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_5_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_5_3_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_5_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_5_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_5_4_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_1_5_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + i*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_1_5_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_1_5_4_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_1_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_1_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_1_4_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_1_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_1_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_1_4_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_1_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_1_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_1_5_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_1_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_1_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_1_5_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_4_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_4_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_4_1_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_4_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_4_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_4_1_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_4_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_4_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_4_5_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_4_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_4_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_4_5_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_5_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_5_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_5_1_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_5_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_5_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_5_1_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_5_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_5_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_5_4_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_3_5_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + k*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_3_5_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_3_5_4_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_1_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_1_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_1_3_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_1_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_1_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_1_3_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_1_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_1_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_1_5_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_1_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_1_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_1_5_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_3_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_3_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_3_1_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_3_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_3_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_3_1_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_3_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_3_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_3_5_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_3_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_3_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_3_5_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_5_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_5_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_5_1_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_5_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_5_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_5_1_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_5_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_5_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_5_3_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_4_5_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + l*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_4_5_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_4_5_3_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_1_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_1_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_1_3_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_1_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_1_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_1_3_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_1_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_1_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_1_4_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_1_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_1_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_1_4_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_3_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_3_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_3_1_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_3_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_3_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_3_1_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_3_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_3_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_3_4_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_3_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_3_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_3_4_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_4_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_4_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_4_1_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_4_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_4_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_4_1_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_4_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_4_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_4_3_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_2_5_4_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = j + m*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_2_5_4_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_2_5_4_3_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_2_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_2_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_2_4_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_2_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_2_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_2_4_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_2_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_2_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_2_5_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_2_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_2_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_2_5_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_4_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_4_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_4_2_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_4_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_4_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_4_2_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_4_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_4_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_4_5_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_4_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_4_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_4_5_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_5_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_5_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_5_2_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_5_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_5_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_5_2_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_5_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_5_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_5_4_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_1_5_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + i*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_1_5_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_1_5_4_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_1_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_1_4_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_1_4_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_1_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_1_4_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_1_4_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_1_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_1_5_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_1_5_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_1_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_1_5_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_1_5_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_4_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_4_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_4_1_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_4_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_4_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_4_1_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_4_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_4_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_4_5_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_4_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + l*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_4_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_4_5_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_5_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_5_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_5_1_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_5_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_5_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_5_1_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_5_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_5_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_5_4_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_2_5_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + j*dimy1 + m*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_2_5_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_2_5_4_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_1_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_1_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_1_2_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_1_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_1_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_1_2_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_1_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_1_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_1_5_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_1_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_1_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_1_5_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_2_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_2_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_2_1_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_2_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_2_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_2_1_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_2_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_2_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_2_5_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_2_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_2_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_2_5_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_5_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_5_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_5_1_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_5_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_5_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_5_1_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_5_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_5_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_5_2_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_4_5_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + l*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_4_5_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_4_5_2_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_1_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_1_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_1_2_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_1_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_1_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_1_2_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_1_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_1_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_1_4_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_1_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_1_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_1_4_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_2_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_2_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_2_1_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_2_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_2_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_2_1_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_2_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_2_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_2_4_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_2_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_2_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_2_4_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_4_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_4_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_4_1_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_4_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_4_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_4_1_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_4_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_4_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_4_2_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_3_5_4_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = k + m*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_3_5_4_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_3_5_4_2_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_2_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_2_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_2_3_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_2_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_2_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_2_3_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_2_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_2_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_2_5_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_2_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_2_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_2_5_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_3_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_3_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_3_2_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_3_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_3_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_3_2_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_3_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_3_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_3_5_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_3_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_3_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_3_5_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_5_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_5_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_5_2_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_5_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_5_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_5_2_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_5_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_5_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_5_3_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_1_5_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + i*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_1_5_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_1_5_3_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_1_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_1_3_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_1_3_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_1_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_1_3_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_1_3_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_1_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_1_5_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_1_5_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_1_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_1_5_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_1_5_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_3_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_3_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_3_1_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_3_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_3_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_3_1_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_3_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_3_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_3_5_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_3_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + k*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_3_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_3_5_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_5_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_5_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_5_1_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_5_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_5_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_5_1_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_5_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_5_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_5_3_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_2_5_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + j*dimy1 + m*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_2_5_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_2_5_3_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_1_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_1_2_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_1_2_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_1_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_1_2_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_1_2_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_1_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_1_5_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_1_5_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_1_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + i*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_1_5_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_1_5_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_2_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_2_1_5_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_2_1_5_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_2_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + m*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_2_1_5_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_2_1_5_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_2_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_2_5_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_2_5_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_2_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + j*dimy1*dimy2 + m*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_2_5_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_2_5_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_5_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_5_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_5_1_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_5_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + m*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_5_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_5_1_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_5_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_5_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_5_2_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_3_5_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + k*dimy1 + m*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_3_5_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_3_5_2_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_1_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_1_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_1_2_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_1_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_1_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_1_2_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_1_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_1_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_1_3_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_1_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_1_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_1_3_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_2_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_2_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_2_1_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_2_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_2_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_2_1_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_2_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_2_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_2_3_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_2_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_2_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_2_3_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_3_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_3_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_3_1_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_3_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_3_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_3_1_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_3_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_3_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_3_2_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_4_5_3_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = l + m*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_4_5_3_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_4_5_3_2_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_2_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_2_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_2_3_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_2_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_2_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_2_3_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_2_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_2_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_2_4_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_2_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_2_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_2_4_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_3_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_3_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_3_2_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_3_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_3_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_3_2_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_3_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_3_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_3_4_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_3_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_3_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_3_4_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_4_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_4_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_4_2_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_4_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_4_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_4_2_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_4_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_4_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_4_3_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_1_4_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + i*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_1_4_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_1_4_3_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_1_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_1_3_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_1_3_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_1_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_1_3_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_1_3_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_1_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_1_4_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_1_4_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_1_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_1_4_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_1_4_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_3_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_3_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_3_1_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_3_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_3_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_3_1_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_3_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_3_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_3_4_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_3_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + k*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_3_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_3_4_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_4_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_4_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_4_1_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_4_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_4_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_4_1_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_4_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_4_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_4_3_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_2_4_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + j*dimy1 + l*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_2_4_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_2_4_3_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_1_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_1_2_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_1_2_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_1_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_1_2_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_1_2_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_1_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_1_4_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_1_4_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_1_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + i*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_1_4_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_1_4_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_2_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_2_1_4_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_2_1_4_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_2_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + l*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_2_1_4_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_2_1_4_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_2_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_2_4_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_2_4_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_2_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + j*dimy1*dimy2 + l*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_2_4_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_2_4_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_4_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_4_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_4_1_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_4_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + l*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_4_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_4_1_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_4_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_4_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_4_2_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_3_4_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + k*dimy1 + l*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_3_4_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_3_4_2_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_1_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_1_2_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_1_2_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_1_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + i*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_1_2_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_1_2_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_1_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_1_3_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_1_3_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_1_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + i*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_1_3_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_1_3_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_2_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_2_1_3_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_2_1_3_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_2_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + j*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + k*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_2_1_3_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_2_1_3_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_2_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_2_3_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_2_3_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_2_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + j*dimy1*dimy2 + k*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_2_3_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_2_3_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_3_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_3_1_2_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_3_1_2_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_3_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + k*dimy1*dimy2 + i*dimy1*dimy2*dimy3 + j*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_3_1_2_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_3_1_2_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_3_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_3_2_1_32(float* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, float* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_3_2_1_32<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _permutedims_5D_5_4_3_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = m + l*dimy1 + k*dimy1*dimy2 + j*dimy1*dimy2*dimy3 + i*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void permutedims_5D_5_4_3_2_1_64(double* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, double* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _permutedims_5D_5_4_3_2_1_64<<<256,256>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
__global__ void _icat_32(int nrows, int ncols, float **x, float *y) {
  int row, col, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    yidx = row + col * nrows;
    y[yidx] = x[col][row];
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void icat_32(int nrows, int ncols, float **x, float *y) {
    float **xx;   
    if (nrows>0 && ncols>0) {
      size_t s = ncols * sizeof(float *);
      cudaMalloc(&xx, s);
      cudaMemcpy(xx, x, s, cudaMemcpyHostToDevice);
      _icat_32<<<256,256>>>(nrows, ncols, xx, y);
      cudaFree(xx);
    }
  }    
}
__global__ void _icat_64(int nrows, int ncols, double **x, double *y) {
  int row, col, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    yidx = row + col * nrows;
    y[yidx] = x[col][row];
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void icat_64(int nrows, int ncols, double **x, double *y) {
    double **xx;   
    if (nrows>0 && ncols>0) {
      size_t s = ncols * sizeof(double *);
      cudaMalloc(&xx, s);
      cudaMemcpy(xx, x, s, cudaMemcpyHostToDevice);
      _icat_64<<<256,256>>>(nrows, ncols, xx, y);
      cudaFree(xx);
    }
  }    
}
static __inline__ __device__ float atomicAdd2(float *address, float val) {
  return atomicAdd(address, val);
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
static __inline__ __device__ double atomicAdd2(double *address, double val) {
  return atomicAdd(address, val);
}
#else      
static __inline__ __device__ double atomicAdd2(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val==0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif
__global__ void _getcols_32(int xrows, int xcols, int ncols, int *cols, float *x, float *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;              
    y[yidx] = x[xidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setcols_32(int xrows, int xcols, int ncols, int *cols, float *x, float *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;              
    x[xidx] = y[yidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _addcols_32(int xrows, int xcols, int ncols, int *cols, float *x, float *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;              
    atomicAdd2(&x[xidx], y[yidx]);
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setcol1_32(int xrows, int xcols, int ncols, int *cols, float *x, float y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;              
    x[xidx] = y;
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _getrows_32(int xrows, int xcols, int nrows, int *rows, float *x, float *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;              
    y[yidx] = x[xidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setrows_32(int xrows, int xcols, int nrows, int *rows, float *x, float *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;              
    x[xidx] = y[yidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _addrows_32(int xrows, int xcols, int nrows, int *rows, float *x, float *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;              
    atomicAdd2(&x[xidx], y[yidx]);
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setrow1_32(int xrows, int xcols, int nrows, int *rows, float *x, float y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;              
    x[xidx] = y;
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _getents_32(int n, int *ents, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = x[ents[i]-1];
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _setents_32(int n, int *ents, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[ents[i]-1] = y[i];
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _addents_32(int n, int *ents, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    atomicAdd2(&x[ents[i]-1], y[i]);
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _setent1_32(int n, int *ents, float *x, float y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[ents[i]-1] = y;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
void getcols_32(int xrows, int xcols, int ncols, int *cols, float *x, float *y)
{ if (ncols>0 && xrows>0 && xcols>0) _getcols_32<<<256,256>>>(xrows,xcols,ncols,cols,x,y); }
void setcols_32(int xrows, int xcols, int ncols, int *cols, float *x, float *y)
{ if (ncols>0 && xrows>0 && xcols>0) _setcols_32<<<256,256>>>(xrows,xcols,ncols,cols,x,y); }
void addcols_32(int xrows, int xcols, int ncols, int *cols, float *x, float *y)
{ if (ncols>0 && xrows>0 && xcols>0) _addcols_32<<<256,256>>>(xrows,xcols,ncols,cols,x,y); }
void setcol1_32(int xrows, int xcols, int ncols, int *cols, float *x, float  y)
{ if (ncols>0 && xrows>0 && xcols>0) _setcol1_32<<<256,256>>>(xrows,xcols,ncols,cols,x,y); }
void getrows_32(int xrows, int xcols, int nrows, int *rows, float *x, float *y)
{ if (nrows>0 && xrows>0 && xcols>0) _getrows_32<<<256,256>>>(xrows,xcols,nrows,rows,x,y); }
void setrows_32(int xrows, int xcols, int nrows, int *rows, float *x, float *y)
{ if (nrows>0 && xrows>0 && xcols>0) _setrows_32<<<256,256>>>(xrows,xcols,nrows,rows,x,y); }
void addrows_32(int xrows, int xcols, int nrows, int *rows, float *x, float *y)
{ if (nrows>0 && xrows>0 && xcols>0) _addrows_32<<<256,256>>>(xrows,xcols,nrows,rows,x,y); }
void setrow1_32(int xrows, int xcols, int nrows, int *rows, float *x, float  y)
{ if (nrows>0 && xrows>0 && xcols>0) _setrow1_32<<<256,256>>>(xrows,xcols,nrows,rows,x,y); }
void getents_32(int n, int *ents, float *x, float *y)
{ if (n>0) _getents_32<<<256,256>>>(n,ents,x,y); }
void setents_32(int n, int *ents, float *x, float *y)
{ if (n>0) _setents_32<<<256,256>>>(n,ents,x,y); }
void addents_32(int n, int *ents, float *x, float *y)
{ if (n>0) _addents_32<<<256,256>>>(n,ents,x,y); }
void setent1_32(int n, int *ents, float *x, float  y)
{ if (n>0) _setent1_32<<<256,256>>>(n,ents,x,y); }
}
__global__ void _getcols_64(int xrows, int xcols, int ncols, int *cols, double *x, double *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;              
    y[yidx] = x[xidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setcols_64(int xrows, int xcols, int ncols, int *cols, double *x, double *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;              
    x[xidx] = y[yidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _addcols_64(int xrows, int xcols, int ncols, int *cols, double *x, double *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;              
    atomicAdd2(&x[xidx], y[yidx]);
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setcol1_64(int xrows, int xcols, int ncols, int *cols, double *x, double y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;              
    x[xidx] = y;
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _getrows_64(int xrows, int xcols, int nrows, int *rows, double *x, double *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;              
    y[yidx] = x[xidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setrows_64(int xrows, int xcols, int nrows, int *rows, double *x, double *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;              
    x[xidx] = y[yidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _addrows_64(int xrows, int xcols, int nrows, int *rows, double *x, double *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;              
    atomicAdd2(&x[xidx], y[yidx]);
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setrow1_64(int xrows, int xcols, int nrows, int *rows, double *x, double y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;              
    x[xidx] = y;
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _getents_64(int n, int *ents, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = x[ents[i]-1];
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _setents_64(int n, int *ents, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[ents[i]-1] = y[i];
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _addents_64(int n, int *ents, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    atomicAdd2(&x[ents[i]-1], y[i]);
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _setent1_64(int n, int *ents, double *x, double y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[ents[i]-1] = y;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
void getcols_64(int xrows, int xcols, int ncols, int *cols, double *x, double *y)
{ if (ncols>0 && xrows>0 && xcols>0) _getcols_64<<<256,256>>>(xrows,xcols,ncols,cols,x,y); }
void setcols_64(int xrows, int xcols, int ncols, int *cols, double *x, double *y)
{ if (ncols>0 && xrows>0 && xcols>0) _setcols_64<<<256,256>>>(xrows,xcols,ncols,cols,x,y); }
void addcols_64(int xrows, int xcols, int ncols, int *cols, double *x, double *y)
{ if (ncols>0 && xrows>0 && xcols>0) _addcols_64<<<256,256>>>(xrows,xcols,ncols,cols,x,y); }
void setcol1_64(int xrows, int xcols, int ncols, int *cols, double *x, double  y)
{ if (ncols>0 && xrows>0 && xcols>0) _setcol1_64<<<256,256>>>(xrows,xcols,ncols,cols,x,y); }
void getrows_64(int xrows, int xcols, int nrows, int *rows, double *x, double *y)
{ if (nrows>0 && xrows>0 && xcols>0) _getrows_64<<<256,256>>>(xrows,xcols,nrows,rows,x,y); }
void setrows_64(int xrows, int xcols, int nrows, int *rows, double *x, double *y)
{ if (nrows>0 && xrows>0 && xcols>0) _setrows_64<<<256,256>>>(xrows,xcols,nrows,rows,x,y); }
void addrows_64(int xrows, int xcols, int nrows, int *rows, double *x, double *y)
{ if (nrows>0 && xrows>0 && xcols>0) _addrows_64<<<256,256>>>(xrows,xcols,nrows,rows,x,y); }
void setrow1_64(int xrows, int xcols, int nrows, int *rows, double *x, double  y)
{ if (nrows>0 && xrows>0 && xcols>0) _setrow1_64<<<256,256>>>(xrows,xcols,nrows,rows,x,y); }
void getents_64(int n, int *ents, double *x, double *y)
{ if (n>0) _getents_64<<<256,256>>>(n,ents,x,y); }
void setents_64(int n, int *ents, double *x, double *y)
{ if (n>0) _setents_64<<<256,256>>>(n,ents,x,y); }
void addents_64(int n, int *ents, double *x, double *y)
{ if (n>0) _addents_64<<<256,256>>>(n,ents,x,y); }
void setent1_64(int n, int *ents, double *x, double  y)
{ if (n>0) _setent1_64<<<256,256>>>(n,ents,x,y); }
}
__global__ void _dropout_32(int n, float p, float q, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] > p) {                  
      y[i] = x[i] * q;
    } else {
      y[i] = 0;
    }
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _dropback_32(int n, float q, float *y, float *dy, float *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] == 0) {
        dx[i] = 0;
    } else {
        dx[i] = dy[i] * q;
    }
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void dropout_32(int n, float p, float *x, float *y) {
    if (n>0) _dropout_32<<<256,256>>>(n,p,1.0/(1.0-p),x,y);
  }    
  void dropback_32(int n, float p, float *x, float *y, float *dy, float *dx) {
    if (n>0) _dropback_32<<<256,256>>>(n,1.0/(1.0-p),y,dy,dx);
  }    
}
__global__ void _dropout_64(int n, double p, double q, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] > p) {                  
      y[i] = x[i] * q;
    } else {
      y[i] = 0;
    }
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _dropback_64(int n, double q, double *y, double *dy, double *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] == 0) {
        dx[i] = 0;
    } else {
        dx[i] = dy[i] * q;
    }
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void dropout_64(int n, double p, double *x, double *y) {
    if (n>0) _dropout_64<<<256,256>>>(n,p,1.0/(1.0-p),x,y);
  }    
  void dropback_64(int n, double p, double *x, double *y, double *dy, double *dx) {
    if (n>0) _dropback_64<<<256,256>>>(n,1.0/(1.0-p),y,dy,dx);
  }    
}
__global__ void _concat_32(int narrays, int *starts, int *lengths, float **x, float *y) {
  int array = blockIdx.x;
  int nelts = lengths[array];                  
  int offset = starts[array];
  for (int i = threadIdx.x; i < nelts; i += blockDim.x) {
    y[i+offset] = x[array][i];
  }
}
extern "C" {
  // julia is responsible for copying args to gpu
  void concat_32(int narrays, int *starts, int *lengths, float **x, float *y) {
    _concat_32<<<narrays,256>>>(narrays, starts, lengths, x, y);
  }    
}
__global__ void _concat_64(int narrays, int *starts, int *lengths, double **x, double *y) {
  int array = blockIdx.x;
  int nelts = lengths[array];                  
  int offset = starts[array];
  for (int i = threadIdx.x; i < nelts; i += blockDim.x) {
    y[i+offset] = x[array][i];
  }
}
extern "C" {
  // julia is responsible for copying args to gpu
  void concat_64(int narrays, int *starts, int *lengths, double **x, double *y) {
    _concat_64<<<narrays,256>>>(narrays, starts, lengths, x, y);
  }    
}
