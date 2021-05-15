# Kernels for elementwise Array,Array,Array->Array ops with equal sized
# arrays used for activation backward functions (with input x,y,dy output dx)

fp = open("cuda111.cu","w")
#using Knet: actback_ops

function cuda111src(f, j=f, ex="$f(xi,yi)"; BLK=256, THR=256)
  sprint() do s
    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,
"""
__global__ void _$(F)_111(int n, $T *x_, $T *y_, $T *dy_, $T *dx_) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    $T x=x_[i];
    $T y=y_[i];
    $T dy=dy_[i];
    dx_[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  $DLLEXPORT void $(F)_111(int n, $T *x, $T *y, $T *dy, $T *dx) {
    _$(F)_111<<<$BLK,$THR>>>(n,x,y,dy,dx);
  }    
  $DLLEXPORT void $(F)_111_stream(int n, $T *x, $T *y, $T *dy, $T *dx, cudaStream_t STR) {
    _$(F)_111<<<$BLK,$THR,0,STR>>>(n,x,y,dy,dx);
  }    
}
""")
    end
  end
end

for a in actback_ops
    if !isa(a,Tuple); a=(a,); end
    print(fp,cuda111src(a...))
end

close(fp)
