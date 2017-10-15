#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _add_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi+yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void add_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _add_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _add_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi+yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void add_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _add_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _add_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi+yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void add_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _add_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _add_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi+yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void add_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _add_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _sub_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi-yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void sub_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _sub_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _sub_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi-yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void sub_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _sub_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _sub_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi-yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void sub_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _sub_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _sub_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi-yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void sub_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _sub_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _mul_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi*yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void mul_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _mul_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _mul_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi*yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void mul_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _mul_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _mul_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi*yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void mul_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _mul_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _mul_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi*yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void mul_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _mul_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _div_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi/yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void div_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _div_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _div_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi/yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void div_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _div_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _div_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi/yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void div_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _div_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _div_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi/yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void div_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _div_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _pow_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=pow(xi,yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void pow_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _pow_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _pow_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=pow(xi,yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void pow_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _pow_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _pow_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=pow(xi,yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void pow_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _pow_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _pow_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=pow(xi,yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void pow_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _pow_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _max_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=(xi>yi?xi:yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void max_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _max_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _max_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=(xi>yi?xi:yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void max_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _max_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _max_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=(xi>yi?xi:yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void max_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _max_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _max_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=(xi>yi?xi:yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void max_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _max_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _min_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=(xi<yi?xi:yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void min_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _min_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _min_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=(xi<yi?xi:yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void min_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _min_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _min_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=(xi<yi?xi:yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void min_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _min_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _min_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=(xi<yi?xi:yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void min_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _min_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _eq_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi==yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void eq_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _eq_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _eq_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi==yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void eq_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _eq_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _eq_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi==yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void eq_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _eq_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _eq_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi==yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void eq_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _eq_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _ne_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi!=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void ne_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _ne_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _ne_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi!=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void ne_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _ne_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _ne_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi!=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void ne_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _ne_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _ne_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi!=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void ne_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _ne_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _gt_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi>yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void gt_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _gt_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _gt_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi>yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void gt_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _gt_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _gt_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi>yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void gt_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _gt_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _gt_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi>yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void gt_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _gt_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _ge_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi>=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void ge_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _ge_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _ge_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi>=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void ge_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _ge_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _ge_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi>=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void ge_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _ge_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _ge_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi>=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void ge_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _ge_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _lt_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi<yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void lt_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _lt_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _lt_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi<yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void lt_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _lt_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _lt_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi<yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void lt_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _lt_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _lt_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi<yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void lt_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _lt_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _le_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=xi<=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void le_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _le_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _le_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=xi<=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void le_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _le_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _le_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=xi<=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void le_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _le_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _le_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=xi<=yi;
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void le_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _le_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _invxback_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=(-xi*yi*yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void invxback_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _invxback_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _invxback_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=(-xi*yi*yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void invxback_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _invxback_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _invxback_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=(-xi*yi*yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void invxback_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _invxback_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _invxback_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=(-xi*yi*yi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void invxback_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _invxback_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _reluback_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=(yi>0?xi:0);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void reluback_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _reluback_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _reluback_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=(yi>0?xi:0);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void reluback_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _reluback_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _reluback_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=(yi>0?xi:0);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void reluback_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _reluback_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _reluback_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=(yi>0?xi:0);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void reluback_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _reluback_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _sigmback_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=(xi*yi*(1-yi));
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void sigmback_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _sigmback_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _sigmback_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=(xi*yi*(1-yi));
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void sigmback_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _sigmback_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _sigmback_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=(xi*yi*(1-yi));
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void sigmback_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _sigmback_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _sigmback_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=(xi*yi*(1-yi));
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void sigmback_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _sigmback_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _tanhback_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=(xi*(1-yi*yi));
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void tanhback_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _tanhback_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _tanhback_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=(xi*(1-yi*yi));
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void tanhback_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _tanhback_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _tanhback_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=(xi*(1-yi*yi));
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void tanhback_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _tanhback_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _tanhback_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=(xi*(1-yi*yi));
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void tanhback_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _tanhback_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 32
#define half_BLOCK_SIZE_y 16
__global__ void _rpow_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float yi = value;
            #else
              float yi = Bs[ty];
            #endif
            z[i]=pow(yi,xi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void rpow_32_13_x_y(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _rpow_32_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _rpow_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double xi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double yi = value;
            #else
              double yi = Bs[ty];
            #endif
            z[i]=pow(yi,xi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void rpow_64_13_x_y(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _rpow_64_13_x_y<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _rpow_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      float value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ float Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            float yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              float xi = value;
            #else
              float xi = Bs[ty];
            #endif
            z[i]=pow(yi,xi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void rpow_32_13_y_x(float *x,float *y,float *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _rpow_32_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
__global__ void _rpow_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    tx+=(ty&1)*32;
    ty=ty>>1;

    #if (__CUDA_ARCH__ >= 300 )
      int laneId = threadIdx.x & 0x1f;
      double value;
      if (laneId == 0)
      {    // all threads except lane 0
          value = y[half_BLOCK_SIZE_y*bx+ty];   // first thread in each wrap loads one element
      }
      value = __shfl(value, 0);   // Get "value" from lane 0

    #else

      __shared__ double Bs[half_BLOCK_SIZE_y];
      if( ty==0 && tx<half_BLOCK_SIZE_y)
      {
        int vector_index = half_BLOCK_SIZE_y*bx+tx;
        Bs[tx]=y[vector_index];
      }
      __syncthreads();

    #endif

    int Start = (((half_BLOCK_SIZE_y*bx)+ty)* brdcastdimstride)+tx;
    int Step = BLOCK_SIZE_x*2;
    if (tx<brdcastdimstride && Start<A_N)
    {
      for (int k=0; k< multidimsize; k++)
      {
        for (int i=Start; i < Start+brdcastdimstride-tx; i+=Step)
        {
            double yi = x[i];
            #if (__CUDA_ARCH__ >= 300 )
              double xi = value;
            #else
              double xi = Bs[ty];
            #endif
            z[i]=pow(yi,xi);
        }
        Start +=brdcastnextstride;
    }
  }
}

extern "C" {
  void rpow_64_13_y_x(double *x,double *y,double *z, int brdcastdimstride, int brdcastnextstride,int multidimsize,int A_N, int B_N) {
    dim3 dimBlock(BLOCK_SIZE_x, BLOCK_SIZE_y);
    int n_block = (B_N+half_BLOCK_SIZE_y-1)/half_BLOCK_SIZE_y;
    dim3 dimGrid(n_block);
    _rpow_64_13_y_x<<<dimGrid,dimBlock>>>(x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,A_N);
  }
}
