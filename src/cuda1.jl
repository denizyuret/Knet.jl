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

function permutedims2Dsrc(f,i1,i2; BLK=256,THR=256)
    sprint() do s
        for (T,F) in [("float","$(f)32"),("double","$(f)64")]
            print(s,
"""
__global__ void _$(F)($T* x, int dimx1, int dimx2, $T* y, int dimy1) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2; v += blockDim.x * gridDim.x) {

    //From 1D to 2D indices
    int i = v % dimx1;
    int j = (v-i) / dimx1;

    //Calculate destination
    int destIndex = $i1 + $i2*dimy1;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void $(F)($T* x, int dimx1, int dimx2, $T* y, int dimy1) {
    _$(F)<<<$BLK,$THR>>>(x,dimx1,dimx2,y,dimy1);
  }    
}
""")
        end
    end
end

function permutedims3Dsrc(f,i1,i2,i3; BLK=256,THR=256)
    sprint() do s
        for (T,F) in [("float","$(f)32"),("double","$(f)64")]
            print(s,
"""
__global__ void _$(F)($T* x, int dimx1, int dimx2, int dimx3, $T* y, int dimy1, int dimy2) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3; v += blockDim.x * gridDim.x) {

    //From 1D to 3D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = (v-i-j*dimx1) / (dimx1*dimx2);

    //Calculate destination
    int destIndex = $i1 + $i2*dimy1 + $i3*dimy1*dimy2;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void $(F)($T* x, int dimx1, int dimx2, int dimx3, $T* y, int dimy1, int dimy2) {
    _$(F)<<<$BLK,$THR>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2);
  }    
}
""")
        end
    end
end

function permutedims4Dsrc(f,i1,i2,i3,i4; BLK=256,THR=256)
    sprint() do s
        for (T,F) in [("float","$(f)32"),("double","$(f)64")]
            print(s,
"""
__global__ void _$(F)($T* x, int dimx1, int dimx2, int dimx3, int dimx4, $T* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4; v += blockDim.x * gridDim.x) {

    //From 1D to 4D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = (v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3);

    //Calculate destination
    int destIndex = $i1 + $i2*dimy1 + $i3*dimy1*dimy2 + $i4*dimy1*dimy2*dimy3;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void $(F)($T* x, int dimx1, int dimx2, int dimx3, int dimx4, $T* y, int dimy1, int dimy2, int dimy3) {
    _$(F)<<<$BLK,$THR>>>(x,dimx1,dimx2,dimx3,dimx4,y,dimy1,dimy2,dimy3);
  }    
}
""")
        end
    end
end

function permutedims5Dsrc(f,i1,i2,i3,i4,i5; BLK=256,THR=256)
    sprint() do s
        for (T,F) in [("float","$(f)32"),("double","$(f)64")]
            print(s,
"""
__global__ void _$(F)($T* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, $T* y, int dimy1, int dimy2, int dimy3, int dimy4) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimx1*dimx2*dimx3*dimx4*dimx5; v += blockDim.x * gridDim.x) {

    //From 1D to 5D indices
    int i = v % dimx1;
    int j = ((v-i) / dimx1) % dimx2;
    int k = ((v-i-j*dimx1) / (dimx1*dimx2)) % dimx3;
    int l = ((v-i-j*dimx1-k*dimx1*dimx2) / (dimx1*dimx2*dimx3)) % dimx4;
    int m = (v-i-j*dimx1-k*dimx1*dimx2-l*dimx1*dimx2*dimx3) / (dimx1*dimx2*dimx3*dimx4);

    //Calculate destination
    int destIndex = $i1 + $i2*dimy1 + $i3*dimy1*dimy2 + $i4*dimy1*dimy2*dimy3 + $i5*dimy1*dimy2*dimy3*dimy4;
    y[destIndex] = x[v];
	}
}
extern "C" {
  void $(F)($T* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, $T* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _$(F)<<<$BLK,$THR>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
""")
        end
    end
end

function cuda1permutedims()
  cudaPerms = [permutedims2Dsrc,permutedims3Dsrc,permutedims4Dsrc,permutedims5Dsrc]
  for i=2:5
      dims = collect(permutations([1:i...],i))
      indnames = collect(permutations(["i","j","k","l","m"][1:i],i))
      for j=1:length(dims)
          fname = string("permutedims_",i,"D",replace(replace(replace(string(dims[j][:]),"[","_"),"]","_"),",","_"))
          print(cudaPerms[i-1](fname,indnames[j]...))
      end
  end
end

cuda1permutedims()