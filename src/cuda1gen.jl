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
""")
        end
    end
end

print(cuda1fill())

function cuda1xfill(; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","32"),("double","64")]
            print(s,
"""
__global__ void _xfill_$F(int nrows, int ncols, $T x, $T *y, int incy) {
  int row, col, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    yidx = row + col * incy;
    y[yidx] = x;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void xfill_$F(int nrows, int ncols, $T x, $T *y, int incy) {
    _xfill_$F<<<$BLK,$THR>>>(nrows, ncols, x, y, incy);
  }    
}
""")
        end
    end
end

print(cuda1xfill())

# copy a block from x into y
function cuda1xcopy(; BLK=256, THR=256)
"""
__global__ void _xcopy(int nrows, int ncols, const char *x, int incx, char *y, int incy) {
  int row, col, xidx, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    xidx = row + col * incx;
    yidx = row + col * incy;
    y[yidx] = x[xidx];
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void xcopy(int nrows, int ncols, const void *x, int incx, void *y, int incy) {
    _xcopy<<<$BLK,$THR>>>(nrows,ncols,(char*)x,incx,(char*)y,incy);
  }    
}
"""
end

print(cuda1xcopy())
