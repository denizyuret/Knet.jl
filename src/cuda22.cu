__device__ void _sum_32_22_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sum_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  float ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=xi; ai=ai+xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _sum_32_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void sum_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  _sum_32_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _sum_64_22_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sum_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  double ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=xi; ai=ai+xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _sum_64_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void sum_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  _sum_64_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _prod_32_22_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai*xi;
}

__global__ void _prod_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  float ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 1;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=xi; ai=ai*xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai*xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _prod_32_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void prod_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  _prod_32_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _prod_64_22_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai*xi;
}

__global__ void _prod_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  double ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 1;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=xi; ai=ai*xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai*xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _prod_64_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void prod_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  _prod_64_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _maximum_32_22_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maximum_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  float ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = (-INFINITY);
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=xi; ai=(ai>xi?ai:xi);
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai>xi?ai:xi);
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _maximum_32_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void maximum_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  _maximum_32_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _maximum_64_22_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maximum_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  double ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = (-INFINITY);
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=xi; ai=(ai>xi?ai:xi);
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai>xi?ai:xi);
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _maximum_64_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void maximum_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  _maximum_64_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _minimum_32_22_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minimum_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  float ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = INFINITY;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=xi; ai=(ai<xi?ai:xi);
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai<xi?ai:xi);
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _minimum_32_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void minimum_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  _minimum_32_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _minimum_64_22_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minimum_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  double ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = INFINITY;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=xi; ai=(ai<xi?ai:xi);
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai<xi?ai:xi);
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _minimum_64_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void minimum_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  _minimum_64_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _sumabs_32_22_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  float ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi<0?-xi:xi); ai=ai+xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _sumabs_32_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void sumabs_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  _sumabs_32_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _sumabs_64_22_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  double ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi<0?-xi:xi); ai=ai+xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _sumabs_64_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void sumabs_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  _sumabs_64_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _sumabs2_32_22_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs2_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  float ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi*xi); ai=ai+xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _sumabs2_32_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void sumabs2_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  _sumabs2_32_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _sumabs2_64_22_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs2_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  double ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi*xi); ai=ai+xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _sumabs2_64_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void sumabs2_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  _sumabs2_64_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _maxabs_32_22_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maxabs_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  float ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi<0?-xi:xi); ai=(ai>xi?ai:xi);
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai>xi?ai:xi);
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _maxabs_32_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void maxabs_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  _maxabs_32_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _maxabs_64_22_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maxabs_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  double ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi<0?-xi:xi); ai=(ai>xi?ai:xi);
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai>xi?ai:xi);
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _maxabs_64_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void maxabs_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  _maxabs_64_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _minabs_32_22_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minabs_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  float ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = INFINITY;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi<0?-xi:xi); ai=(ai<xi?ai:xi);
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai<xi?ai:xi);
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _minabs_32_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void minabs_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  _minabs_32_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _minabs_64_22_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minabs_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  double ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = INFINITY;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi<0?-xi:xi); ai=(ai<xi?ai:xi);
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai<xi?ai:xi);
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _minabs_64_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void minabs_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  _minabs_64_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _countnz_32_22_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _countnz_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  float ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi!=0); ai=ai+xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _countnz_32_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void countnz_32_22(int nx, int xd1, float *x, int s1, int s2, int ny, float *y) {
  _countnz_32_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

__device__ void _countnz_64_22_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _countnz_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[128];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int d = blockDim.x;
  double ai, xi;

  for (int bi = b; bi < ny; bi += blockDim.x) {
    // sum the elements assigned to this thread
    ai = 0;
    int lower = (bi%s1) + (bi/s1)*s2;
    int upper = lower + s1*xd1;
    for (int i=lower+t*s1; i <= upper; i+=d*s1) {
      xi = x[i]; xi=(xi!=0); ai=ai+xi;
    }
    buffer[t] = ai;
    __syncthreads();

    // help sum the entries in the block
    for(int stride=128/2; stride>32; stride>>=1) {
      if(t < stride) {
        ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
      }
      __syncthreads();   // Q: can this be outside the for loop?
    }

    if(t<32) {
      _countnz_64_22_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
    }
    __syncthreads();

    if(t==0) {  // the first thread in the block writes the block result to y
      y[bi]=buffer[0];
    }
  }
}

extern "C" { void countnz_64_22(int nx, int xd1, double *x, int s1, int s2, int ny, double *y) {
  _countnz_64_22<<<128,128>>>(nx,xd1,x,s1,s2,ny,y);
}}

