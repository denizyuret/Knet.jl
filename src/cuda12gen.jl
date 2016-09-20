using Knet: cuda12

# Full broadcasting index computation is complicated.  I am going to
# handle the case where the input arrays are either the same size as
# the result or vectors (have a single dim > 1).  If the result array
# has size (a1,a2,...) and input array b has size (1,...,bn,...,1), we
# need to find the linear index in b given the linear index in a.  If
# we are at position (i1,i2,...) the answer is just
# i_n=mod(div(i,stride(a,n)),size(a,n)) with 0 indexing.  So we can
# just pass in stride(a,n) and size(a,n) as an argument for each
# input.

function cuda12src(f, j=f, ex="$f(xi,yi)"; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
            print(s,
"""
__global__ void _$(F)_12(int n, $T *x, int sx, int nx, $T *y, int sy, int ny, $T *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    $T xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    $T yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $(F)_12(int n, $T *x, int sx, int nx, $T *y, int sy, int ny, $T *z) {
    _$(F)_12<<<$BLK,$THR>>>(n,x,sx,nx,y,sy,ny,z);
  }    
}
""")
        end
    end
end

for a in cuda12
    isa(a,Tuple) || (a=(a,))
    print(cuda12src(a...))
end
