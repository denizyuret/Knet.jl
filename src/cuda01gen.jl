using Knet: cuda01

function cuda01src(f, j=f, ex="$f(s,xi)"; BLK=256, THR=256)
  sprint() do s
    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,
"""
__global__ void _$(F)_01(int n, $T s, $T *x, $T *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    $T xi = x[i];
    y[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $(F)_01(int n, $T s, $T *x, $T *y) {
    _$(F)_01<<<$BLK,$THR>>>(n,s,x,y);
  }    
}
""")
    end
  end
end

for a in cuda01
    isa(a,Tuple) || (a=(a,))
    print(cuda01src(a...))
end
