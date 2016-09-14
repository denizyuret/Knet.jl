using Knet: cuda1

function cuda1src(f, j=f, ex="$f(xi)"; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
            print(s,
"""
__global__ void _$F(int n, $T *x, $T *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    $T xi = x[i];
    y[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void $F(int n, $T *x, $T *y) {
    _$F<<<$BLK,$THR>>>(n,x,y);
  }    
}
""")
        end
    end
end

for a in cuda1
    isa(a,Tuple) || (a=(a,))
    print(cuda1src(a...))
end


function cuda1fill(; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","32"),("double","64")]
            print(s,
"""
__global__ void _fill_$F(int n, $T x, $T *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = x;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void fill_$F(int n, $T x, $T *y) {
    _fill_$F<<<$BLK,$THR>>>(n,x,y);
  }    
}
__global__ void _xfill_$F(int xlen, int xrows, int yidx0, int yrows, $T x, $T *y) {
  int xrow, xcol, yidx;
  int xidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (xidx < xlen) {
    xrow = xidx % xrows;
    xcol = xidx / xrows;
    yidx = yidx0 + yrows * xcol + xrow;
    y[yidx] = x;
    xidx += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void xfill_$F(int xlen, int xrows, int yidx0, int yrows, $T x, $T *y) {
    _xfill_$F<<<$BLK,$THR>>>(xlen,xrows,yidx0,yrows,x,y);
  }    
}
""")
        end
    end
end

print(cuda1fill())

# copy x into a particular position in y
function cuda1copy(; BLK=256, THR=256)
"""
__global__ void _xcopy(int xlen, int xrows, int yidx0, int yrows, const char *x, char *y) {
  int xrow, xcol, yidx;
  int xidx = threadIdx.x + blockIdx.x * blockDim.x;
  while (xidx < xlen) {
    xrow = xidx % xrows;
    xcol = xidx / xrows;
    yidx = yidx0 + yrows * xcol + xrow;
    y[yidx] = x[xidx];
    xidx += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void xcopy(int xlen, int xrows, int yidx0, int yrows, const void *x, void *y) {
    _xcopy<<<$BLK,$THR>>>(xlen,xrows,yidx0,yrows,(char*)x,(char*)y);
  }    
}
"""
end

print(cuda1copy())
