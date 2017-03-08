# Kernels for unary array operations

using Knet: unary_ops

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

for a in unary_ops
    if !isa(a,Tuple); a=(a,); end
    print(cuda1src(a...))
end

# Kernels used by setindex! and getindex: fill, xfill, xcopy:

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


permutedims3D_ops = [
#("permutedims3D_1_2_3","i","j","k"),#noop
("permutedims3D_1_3_2","i","k","j"),
("permutedims3D_2_1_3","j","i","k"),
("permutedims3D_2_3_1","k","i","j"),
("permutedims3D_3_1_2","j","k","i"),
("permutedims3D_3_2_1","k","j","i"),
]

function permutedims3Dsrc(f, i1, i2, i3; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
            print(s,
"""
__global__ void _$(F)_44($T* x, int dimx1, int dimx2, int dimx3, $T* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = $i1 + dimx1*$i2 + dimx1*dimx2*$i3;
		y[v] = x[srcIndex];
	}
}
extern "C" {
  void $(F)_44($T* x, int dimx1, int dimx2, int dimx3, $T* y, int dimy1, int dimy2, int dimy3) {
    _$(F)_44<<<$BLK,$THR>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }    
}
""")
        end
    end
end

for a in permutedims3D_ops
    if !isa(a,Tuple); a=(a,); end
    print(permutedims3Dsrc(a...))
end

function cuda1icat(; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","32"),("double","64")]
            print(s,
"""
__global__ void _icat_$F(int nrows, int ncols, $T **x, $T *y) {
  int row, col, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    yidx = row + col * nrows;
    y[yidx] = x[col][row];
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void icat_$F(int nrows, int ncols, $T **x, $T *y) {
    $T **xx;   
    size_t s = ncols * sizeof($T *);
    cudaMalloc(&xx, s);
    cudaMemcpy(xx, x, s, cudaMemcpyHostToDevice);
    _icat_$F<<<$BLK,$THR>>>(nrows, ncols, xx, y);
    cudaFree(xx);
  }    
}
""")
        end
    end
end

print(cuda1icat())
