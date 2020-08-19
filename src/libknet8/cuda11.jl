# Kernels for elementwise Array,Array->Array ops with equal sized
# arrays.

fp = open("cuda11.cu","w")
#using Knet: binary_ops

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
  $DLLEXPORT void $(F)_11(int n, $T *x, $T *y, $T *z) {
    _$(F)_11<<<$BLK,$THR>>>(n,x,y,z);
  }    
}
""")
    end
  end
end

for a in binary_ops
    if !isa(a,Tuple); a=(a,); end
    print(fp,cuda11src(a...))
end

close(fp)
