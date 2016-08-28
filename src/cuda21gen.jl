using Knet: cuda21

# Reduction to a vector

function cuda21src(f, j, op, f1, v0; BLK=128, THR=128) # BLK not used, determined by ny
    sprint() do s
        for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
            print(s,
"""
__device__ void _$(F)_21_0(volatile $T *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  $T ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=$op;
  ai=x[i]; xi=x[i+16]; x[i]=$op;
  ai=x[i]; xi=x[i+ 8]; x[i]=$op;
  ai=x[i]; xi=x[i+ 4]; x[i]=$op;
  ai=x[i]; xi=x[i+ 2]; x[i]=$op;
  ai=x[i]; xi=x[i+ 1]; x[i]=$op;
}

__global__ void _$(F)_21(int nx, $T *x, int sy, int ny, $T *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ $T buffer[$THR];
  int t = threadIdx.x;
  int b = blockIdx.x;
  $T ai, xi;

  // sum the elements assigned to this thread
  ai = $v0;
  if (sy == 1) {
     int istep = $THR*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=$f1; ai=$op;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=$THR) {
        xi=x[i]; xi=$f1; ai=$op;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=$THR/2; stride>32; stride>>=1) { 
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=$op;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {  
    _$(F)_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

extern "C" { void $(F)_21(int nx, $T *x, int sy, int ny, $T *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _$(F)_21<<<$BLK,$THR>>>(nx,x,sy,ny,y);
  _$(F)_21<<<ny,$THR>>>(nx,x,sy,ny,y);
}}

""")
        end
    end
end

for a in cuda21
    isa(a,Tuple) || (a=(a,))
    print(cuda21src(a...))
end
