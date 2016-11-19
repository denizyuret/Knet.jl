using Knet: cuda11

function cuda11src(f, j=f, ex="$f(xi,yi)"; BLK=256, THR=256)
  sprint() do s
    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,
"""
__global__ void _$(F)_11(int n, $T *x, $T *y, $T *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    $T xi=x[i];
    $T yi=y[i];
    z[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $(F)_11(int n, $T *x, $T *y, $T *z) {
    _$(F)_11<<<$BLK,$THR>>>(n,x,y,z);
  }    
}
""")
    end
  end
end

for a in cuda11
    isa(a,Tuple) || (a=(a,))
    print(cuda11src(a...))
end
