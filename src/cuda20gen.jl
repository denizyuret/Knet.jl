using Knet: cuda20

# Reduction to a scalar
# CUBLAS nrm2 is extremely slow.  The following is based on code from Barret Zoph.
# Based on: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf

function cuda20src(f, j, op, f1, v0; BLK=128, THR=128)
    sprint() do s
        for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
            print(s,
"""
__device__ void _$(F)_20_0(volatile $T *x, int i) {
//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
  $T ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=$op;
  ai=x[i]; xi=x[i+16]; x[i]=$op;
  ai=x[i]; xi=x[i+ 8]; x[i]=$op;
  ai=x[i]; xi=x[i+ 4]; x[i]=$op;
  ai=x[i]; xi=x[i+ 2]; x[i]=$op;
  ai=x[i]; xi=x[i+ 1]; x[i]=$op;
}

__global__ void _$(F)_20_1(int n, $T *x, $T *y) {
  __shared__ $T buffer[$THR];			   //all THR threads in the block write to buffer on their own tid
  int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
  int i_end = n; //end at dim
  int i_step = blockDim.x*gridDim.x; // step is the total number of threads in the system
  int tid = threadIdx.x;
  $T ai, xi;

  // sum the elements assigned to this thread
  ai = $v0;
  for(int i=i_start; i<i_end; i+=i_step) {
     xi=x[i]; xi=$f1; ai=$op;
  }
  buffer[tid] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=$THR/2; stride>32; stride>>=1) { 
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=$op;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(tid<32) {  
    _$(F)_20_0(buffer,tid);  // Inlining this does not work.
  }
  __syncthreads();

  if(tid==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

__global__ void _$(F)_20_2($T *y,$T *z) {   // sum block results in y
  __shared__ $T buffer[$BLK];
  $T ai, xi;
  int tid = threadIdx.x;
  buffer[tid] = y[tid];
  __syncthreads();
  for(int stride=$BLK/2; stride>32; stride>>=1) {
    if(tid < stride) {
      ai=buffer[tid]; xi=buffer[stride+tid]; buffer[tid]=$op;
    }
    __syncthreads();
  }
  if(tid<32) {
    _$(F)_20_0(buffer,tid);
  }
  __syncthreads();
  if(tid==0) {
    z[0]=buffer[0];
  }
}

extern "C" { $T $(F)_20(int n, $T *x) {
  $T r;
  static $T *y;
  static $T *z;
  if (y == NULL) cudaMalloc(&y, $BLK*sizeof($T)); // sum for each block
  if (z == NULL) cudaMalloc(&z, sizeof($T));      // final sum
  _$(F)_20_1<<<$BLK,$THR>>>(n,x,y);
  _$(F)_20_2<<<1,$BLK>>>(y,z);                  
  cudaMemcpy(&r,z,sizeof($T),cudaMemcpyDeviceToHost);
  return r;
}}

""")
        end
    end
end

for a in cuda20
    isa(a,Tuple) || (a=(a,))
    print(cuda20src(a...))
end
