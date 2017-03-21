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
  if (nrows == 0) return;
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
  if (nrows == 0) return;
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
  if (nrows == 0) return;
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

# This is for missing double atomicAdd()
print("""
static __inline__ __device__ float atomicAdd2(float *address, float val) {
  return atomicAdd(address, val);
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
static __inline__ __device__ double atomicAdd2(double *address, double val) {
  return atomicAdd(address, val);
}
#else
static __inline__ __device__ double atomicAdd2(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val==0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif
""")

function cuda1getcols(; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","32"),("double","64")]
            print(s,
"""
__global__ void _getcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (xrows == 0) return;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;
    y[yidx] = x[xidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (xrows == 0) return;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;
    x[xidx] = y[yidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _addcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (xrows == 0) return;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;
    atomicAdd2(&x[xidx], y[yidx]);
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setcol1_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (xrows == 0) return;
  while (1) {
    row = yidx % xrows;
    col = yidx / xrows;
    if (col >= ncols) break;
    xidx = row + (cols[col]-1) * xrows;
    x[xidx] = y;
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _getrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (nrows == 0) return;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;
    y[yidx] = x[xidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (nrows == 0) return;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;
    x[xidx] = y[yidx];
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _addrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (nrows == 0) return;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;
    atomicAdd2(&x[xidx], y[yidx]);
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _setrow1_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T y) {
  int row, col, xidx;
  int yidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (nrows == 0) return;
  while (1) {
    row = yidx % nrows;
    col = yidx / nrows;
    if (col >= xcols) break;
    xidx = rows[row] - 1 + col * xrows;
    x[xidx] = y;
    yidx += blockDim.x * gridDim.x;
  }
}
__global__ void _getents_$F(int n, int *ents, $T *x, $T *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = x[ents[i]-1];
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _setents_$F(int n, int *ents, $T *x, $T *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[ents[i]-1] = y[i];
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _addents_$F(int n, int *ents, $T *x, $T *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    atomicAdd2(&x[ents[i]-1], y[i]);
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _setent1_$F(int n, int *ents, $T *x, $T y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    x[ents[i]-1] = y;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
void getcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y) { _getcols_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
void setcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y) { _setcols_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
void addcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y) { _addcols_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
void setcol1_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T  y) { _setcol1_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
void getrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y) { _getrows_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
void setrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y) { _setrows_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
void addrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y) { _addrows_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
void setrow1_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T  y) { _setrow1_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
void getents_$F(int n, int *ents, $T *x, $T *y) { _getents_$F<<<$BLK,$THR>>>(n,ents,x,y); }
void setents_$F(int n, int *ents, $T *x, $T *y) { _setents_$F<<<$BLK,$THR>>>(n,ents,x,y); }
void addents_$F(int n, int *ents, $T *x, $T *y) { _addents_$F<<<$BLK,$THR>>>(n,ents,x,y); }
void setent1_$F(int n, int *ents, $T *x, $T  y) { _setent1_$F<<<$BLK,$THR>>>(n,ents,x,y); }
}
""")
        end
    end
end

print(cuda1getcols())


# Dropout

function cuda1dropout(; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","32"),("double","64")]
            print(s,
"""
__global__ void _dropout_$F(int n, $T p, $T q, $T *x, $T *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] > p) {
      y[i] = x[i] * q;
    } else {
      y[i] = 0;
    }
    i += blockDim.x * gridDim.x;
  }
}
__global__ void _dropback_$F(int n, $T q, $T *y, $T *dy, $T *dx) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (y[i] == 0) {
        dx[i] = 0;
    } else {
        dx[i] = dy[i] * q;
    }
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void dropout_$F(int n, $T p, $T *x, $T *y) {
    _dropout_$F<<<$BLK,$THR>>>(n,p,1.0/(1.0-p),x,y);
  }
  void dropback_$F(int n, $T p, $T *x, $T *y, $T *dy, $T *dx) {
    _dropback_$F<<<$BLK,$THR>>>(n,1.0/(1.0-p),y,dy,dx);
  }
}
""")
        end
    end
end

print(cuda1dropout())
