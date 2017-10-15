__device__ void _sum_32_20_0(volatile float *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sum_32_20_1(int n, float *x, float *y) {
  __shared__ float buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=xi; ai=ai+xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _sum_32_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _sum_32_20_2(float *y,float *z) {   // sum block results in y
  __shared__ float buffer[128];
  float ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _sum_32_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { float sum_32_20(int n, float *x) {
  float r;
  static float *y;
  static float *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(float)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(float));      // final sum
  _sum_32_20_1<<<128,128>>>(n,x,y);
  _sum_32_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(float),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _sum_64_20_0(volatile double *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sum_64_20_1(int n, double *x, double *y) {
  __shared__ double buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=xi; ai=ai+xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _sum_64_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _sum_64_20_2(double *y,double *z) {   // sum block results in y
  __shared__ double buffer[128];
  double ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _sum_64_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { double sum_64_20(int n, double *x) {
  double r;
  static double *y;
  static double *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(double)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(double));      // final sum
  _sum_64_20_1<<<128,128>>>(n,x,y);
  _sum_64_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(double),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _prod_32_20_0(volatile float *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai*xi;
}

__global__ void _prod_32_20_1(int n, float *x, float *y) {
  __shared__ float buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 1;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=xi; ai=ai*xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai*xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _prod_32_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _prod_32_20_2(float *y,float *z) {   // sum block results in y
  __shared__ float buffer[128];
  float ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai*xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _prod_32_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { float prod_32_20(int n, float *x) {
  float r;
  static float *y;
  static float *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(float)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(float));      // final sum
  _prod_32_20_1<<<128,128>>>(n,x,y);
  _prod_32_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(float),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _prod_64_20_0(volatile double *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai*xi;
}

__global__ void _prod_64_20_1(int n, double *x, double *y) {
  __shared__ double buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 1;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=xi; ai=ai*xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai*xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _prod_64_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _prod_64_20_2(double *y,double *z) {   // sum block results in y
  __shared__ double buffer[128];
  double ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai*xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _prod_64_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { double prod_64_20(int n, double *x) {
  double r;
  static double *y;
  static double *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(double)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(double));      // final sum
  _prod_64_20_1<<<128,128>>>(n,x,y);
  _prod_64_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(double),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _maximum_32_20_0(volatile float *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maximum_32_20_1(int n, float *x, float *y) {
  __shared__ float buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = (-INFINITY);
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=xi; ai=(ai>xi?ai:xi);
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai>xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _maximum_32_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _maximum_32_20_2(float *y,float *z) {   // sum block results in y
  __shared__ float buffer[128];
  float ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai>xi?ai:xi);
    }
    __syncthreads();
  }
  if(tid<32) {
    _maximum_32_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { float maximum_32_20(int n, float *x) {
  float r;
  static float *y;
  static float *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(float)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(float));      // final sum
  _maximum_32_20_1<<<128,128>>>(n,x,y);
  _maximum_32_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(float),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _maximum_64_20_0(volatile double *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maximum_64_20_1(int n, double *x, double *y) {
  __shared__ double buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = (-INFINITY);
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=xi; ai=(ai>xi?ai:xi);
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai>xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _maximum_64_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _maximum_64_20_2(double *y,double *z) {   // sum block results in y
  __shared__ double buffer[128];
  double ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai>xi?ai:xi);
    }
    __syncthreads();
  }
  if(tid<32) {
    _maximum_64_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { double maximum_64_20(int n, double *x) {
  double r;
  static double *y;
  static double *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(double)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(double));      // final sum
  _maximum_64_20_1<<<128,128>>>(n,x,y);
  _maximum_64_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(double),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _minimum_32_20_0(volatile float *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minimum_32_20_1(int n, float *x, float *y) {
  __shared__ float buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = INFINITY;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=xi; ai=(ai<xi?ai:xi);
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai<xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _minimum_32_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _minimum_32_20_2(float *y,float *z) {   // sum block results in y
  __shared__ float buffer[128];
  float ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai<xi?ai:xi);
    }
    __syncthreads();
  }
  if(tid<32) {
    _minimum_32_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { float minimum_32_20(int n, float *x) {
  float r;
  static float *y;
  static float *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(float)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(float));      // final sum
  _minimum_32_20_1<<<128,128>>>(n,x,y);
  _minimum_32_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(float),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _minimum_64_20_0(volatile double *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minimum_64_20_1(int n, double *x, double *y) {
  __shared__ double buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = INFINITY;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=xi; ai=(ai<xi?ai:xi);
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai<xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _minimum_64_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _minimum_64_20_2(double *y,double *z) {   // sum block results in y
  __shared__ double buffer[128];
  double ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai<xi?ai:xi);
    }
    __syncthreads();
  }
  if(tid<32) {
    _minimum_64_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { double minimum_64_20(int n, double *x) {
  double r;
  static double *y;
  static double *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(double)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(double));      // final sum
  _minimum_64_20_1<<<128,128>>>(n,x,y);
  _minimum_64_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(double),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _sumabs_32_20_0(volatile float *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs_32_20_1(int n, float *x, float *y) {
  __shared__ float buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi<0?-xi:xi); ai=ai+xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _sumabs_32_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _sumabs_32_20_2(float *y,float *z) {   // sum block results in y
  __shared__ float buffer[128];
  float ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _sumabs_32_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { float sumabs_32_20(int n, float *x) {
  float r;
  static float *y;
  static float *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(float)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(float));      // final sum
  _sumabs_32_20_1<<<128,128>>>(n,x,y);
  _sumabs_32_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(float),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _sumabs_64_20_0(volatile double *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs_64_20_1(int n, double *x, double *y) {
  __shared__ double buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi<0?-xi:xi); ai=ai+xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _sumabs_64_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _sumabs_64_20_2(double *y,double *z) {   // sum block results in y
  __shared__ double buffer[128];
  double ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _sumabs_64_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { double sumabs_64_20(int n, double *x) {
  double r;
  static double *y;
  static double *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(double)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(double));      // final sum
  _sumabs_64_20_1<<<128,128>>>(n,x,y);
  _sumabs_64_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(double),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _sumabs2_32_20_0(volatile float *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs2_32_20_1(int n, float *x, float *y) {
  __shared__ float buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi*xi); ai=ai+xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _sumabs2_32_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _sumabs2_32_20_2(float *y,float *z) {   // sum block results in y
  __shared__ float buffer[128];
  float ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _sumabs2_32_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { float sumabs2_32_20(int n, float *x) {
  float r;
  static float *y;
  static float *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(float)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(float));      // final sum
  _sumabs2_32_20_1<<<128,128>>>(n,x,y);
  _sumabs2_32_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(float),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _sumabs2_64_20_0(volatile double *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs2_64_20_1(int n, double *x, double *y) {
  __shared__ double buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi*xi); ai=ai+xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _sumabs2_64_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _sumabs2_64_20_2(double *y,double *z) {   // sum block results in y
  __shared__ double buffer[128];
  double ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _sumabs2_64_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { double sumabs2_64_20(int n, double *x) {
  double r;
  static double *y;
  static double *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(double)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(double));      // final sum
  _sumabs2_64_20_1<<<128,128>>>(n,x,y);
  _sumabs2_64_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(double),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _maxabs_32_20_0(volatile float *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maxabs_32_20_1(int n, float *x, float *y) {
  __shared__ float buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi<0?-xi:xi); ai=(ai>xi?ai:xi);
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai>xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _maxabs_32_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _maxabs_32_20_2(float *y,float *z) {   // sum block results in y
  __shared__ float buffer[128];
  float ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai>xi?ai:xi);
    }
    __syncthreads();
  }
  if(tid<32) {
    _maxabs_32_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { float maxabs_32_20(int n, float *x) {
  float r;
  static float *y;
  static float *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(float)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(float));      // final sum
  _maxabs_32_20_1<<<128,128>>>(n,x,y);
  _maxabs_32_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(float),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _maxabs_64_20_0(volatile double *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maxabs_64_20_1(int n, double *x, double *y) {
  __shared__ double buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi<0?-xi:xi); ai=(ai>xi?ai:xi);
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai>xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _maxabs_64_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _maxabs_64_20_2(double *y,double *z) {   // sum block results in y
  __shared__ double buffer[128];
  double ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai>xi?ai:xi);
    }
    __syncthreads();
  }
  if(tid<32) {
    _maxabs_64_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { double maxabs_64_20(int n, double *x) {
  double r;
  static double *y;
  static double *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(double)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(double));      // final sum
  _maxabs_64_20_1<<<128,128>>>(n,x,y);
  _maxabs_64_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(double),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _minabs_32_20_0(volatile float *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minabs_32_20_1(int n, float *x, float *y) {
  __shared__ float buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = INFINITY;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi<0?-xi:xi); ai=(ai<xi?ai:xi);
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai<xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _minabs_32_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _minabs_32_20_2(float *y,float *z) {   // sum block results in y
  __shared__ float buffer[128];
  float ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai<xi?ai:xi);
    }
    __syncthreads();
  }
  if(tid<32) {
    _minabs_32_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { float minabs_32_20(int n, float *x) {
  float r;
  static float *y;
  static float *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(float)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(float));      // final sum
  _minabs_32_20_1<<<128,128>>>(n,x,y);
  _minabs_32_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(float),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _minabs_64_20_0(volatile double *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minabs_64_20_1(int n, double *x, double *y) {
  __shared__ double buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = INFINITY;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi<0?-xi:xi); ai=(ai<xi?ai:xi);
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai<xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _minabs_64_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _minabs_64_20_2(double *y,double *z) {   // sum block results in y
  __shared__ double buffer[128];
  double ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=(ai<xi?ai:xi);
    }
    __syncthreads();
  }
  if(tid<32) {
    _minabs_64_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { double minabs_64_20(int n, double *x) {
  double r;
  static double *y;
  static double *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(double)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(double));      // final sum
  _minabs_64_20_1<<<128,128>>>(n,x,y);
  _minabs_64_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(double),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _countnz_32_20_0(volatile float *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _countnz_32_20_1(int n, float *x, float *y) {
  __shared__ float buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi!=0); ai=ai+xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _countnz_32_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _countnz_32_20_2(float *y,float *z) {   // sum block results in y
  __shared__ float buffer[128];
  float ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _countnz_32_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { float countnz_32_20(int n, float *x) {
  float r;
  static float *y;
  static float *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(float)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(float));      // final sum
  _countnz_32_20_1<<<128,128>>>(n,x,y);
  _countnz_32_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(float),cudaMemcpyDeviceToHost);
  return r;
}}

__device__ void _countnz_64_20_0(volatile double *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _countnz_64_20_1(int n, double *x, double *y) {
  __shared__ double buffer[128];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=(xi!=0); ai=ai+xi;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=128/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _countnz_64_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _countnz_64_20_2(double *y,double *z) {   // sum block results in y
  __shared__ double buffer[128];
  double ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=128/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=ai+xi;
    }
    __syncthreads();
  }
  if(tid<32) {
    _countnz_64_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { double countnz_64_20(int n, double *x) {
  double r;
  static double *y;
  static double *z;
  if (y == NULL) cudaMalloc(&y, 128*sizeof(double)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof(double));      // final sum
  _countnz_64_20_1<<<128,128>>>(n,x,y);
  _countnz_64_20_2<<<1,128>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof(double),cudaMemcpyDeviceToHost);
  return r;
}}

