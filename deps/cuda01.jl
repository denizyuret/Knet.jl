# Kernels for Scalar,Array->Array

fp = open("cuda01.cu","w")
using Knet: broadcast_ops

function cuda01src(f, j=f, ex="$f(xi,yi)"; BLK=256, THR=256)
  sprint() do s
    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,
"""
__global__ void _$(F)_01(int n, $T xi, $T *y, $T *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    $T yi = y[i];
    z[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  $DLLEXPORT void $(F)_01(int n, $T xi, $T *y, $T *z) {
    _$(F)_01<<<$BLK,$THR>>>(n,xi,y,z);
  }    
}
""")
    end
  end
end

for a in broadcast_ops
    if !isa(a,Tuple); a=(a,); end
    print(fp, cuda01src(a...))
end
close(fp)
