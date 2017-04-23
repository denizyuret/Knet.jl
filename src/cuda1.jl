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
    if (n>0) _$F<<<$BLK,$THR>>>(n,x,y);
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
    if (n>0) _fill_$F<<<$BLK,$THR>>>(n,x,y);
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
    if (nrows>0 && ncols>0) _xfill_$F<<<$BLK,$THR>>>(nrows, ncols, x, y, incy);
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
    if (nrows>0 && ncols>0) _xcopy<<<$BLK,$THR>>>(nrows,ncols,(char*)x,incx,(char*)y,incy);
  }    
}
"""
end

print(cuda1xcopy())


### Kernels for permutedims by Ekrem Emre Yurdakul 2017-02-27

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

using Combinatorics

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
    if (nrows>0 && ncols>0) {
      size_t s = ncols * sizeof($T *);
      cudaMalloc(&xx, s);
      cudaMemcpy(xx, x, s, cudaMemcpyHostToDevice);
      _icat_$F<<<$BLK,$THR>>>(nrows, ncols, xx, y);
      cudaFree(xx);
    }
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
void getcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y)
{ if (ncols>0 && xrows>0 && xcols>0) _getcols_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
void setcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y)
{ if (ncols>0 && xrows>0 && xcols>0) _setcols_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
void addcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y)
{ if (ncols>0 && xrows>0 && xcols>0) _addcols_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
void setcol1_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T  y)
{ if (ncols>0 && xrows>0 && xcols>0) _setcol1_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
void getrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y)
{ if (nrows>0 && xrows>0 && xcols>0) _getrows_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
void setrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y)
{ if (nrows>0 && xrows>0 && xcols>0) _setrows_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
void addrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y)
{ if (nrows>0 && xrows>0 && xcols>0) _addrows_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
void setrow1_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T  y)
{ if (nrows>0 && xrows>0 && xcols>0) _setrow1_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
void getents_$F(int n, int *ents, $T *x, $T *y)
{ if (n>0) _getents_$F<<<$BLK,$THR>>>(n,ents,x,y); }
void setents_$F(int n, int *ents, $T *x, $T *y)
{ if (n>0) _setents_$F<<<$BLK,$THR>>>(n,ents,x,y); }
void addents_$F(int n, int *ents, $T *x, $T *y)
{ if (n>0) _addents_$F<<<$BLK,$THR>>>(n,ents,x,y); }
void setent1_$F(int n, int *ents, $T *x, $T  y)
{ if (n>0) _setent1_$F<<<$BLK,$THR>>>(n,ents,x,y); }
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
    if (n>0) _dropout_$F<<<$BLK,$THR>>>(n,p,1.0/(1.0-p),x,y);
  }    
  void dropback_$F(int n, $T p, $T *x, $T *y, $T *dy, $T *dx) {
    if (n>0) _dropback_$F<<<$BLK,$THR>>>(n,1.0/(1.0-p),y,dy,dx);
  }    
}
""")
        end
    end
end

print(cuda1dropout())

# This is still too slow compared to concat on cpu and copy to gpu
# Tested for 25 arrays of 200
function cuda1concat(; BLK=256, THR=256)
    sprint() do s
        for (T,F) in [("float","32"),("double","64")]
            print(s,
"""
__global__ void _concat_$F(int narrays, int *starts, int *lengths, $T **x, $T *y) {
  int array = blockIdx.x;
  int nelts = lengths[array];                  
  int offset = starts[array];
  for (int i = threadIdx.x; i < nelts; i += blockDim.x) {
    y[i+offset] = x[array][i];
  }
}
extern "C" {
  // julia is responsible for copying args to gpu
  void concat_$F(int narrays, int *starts, int *lengths, $T **x, $T *y) {
    _concat_$F<<<narrays,$THR>>>(narrays, starts, lengths, x, y);
  }    
}
""")
        end
    end
end

print(cuda1concat())

# Here is the test script for cuda1concat:
# using Knet, BenchmarkTools
# using Knet: @knet8

# for S in (32,64); T = Symbol("Float$S"); F = "concat_$S"
# @eval function concat(A::KnetArray{$T}...)
#     nargs = length(A)
#     S = Array(Int32, nargs)
#     L = Array(Int32, nargs)
#     nelts = 0
#     @inbounds for i in 1:nargs
#         n = length(A[i])
#         S[i] = nelts
#         L[i] = n
#         nelts += n
#     end
#     S = KnetArray(S)
#     L = KnetArray(L)
#     X = KnetArray([map(pointer,A)...])
#     Y = KnetArray{$T}(nelts)
#     @knet8($F,(Cint,Ptr{Cint},Ptr{Cint},Ptr{Ptr{$T}},Ptr{$T}),nargs,S,L,X,Y)
#     return Y
# end
# end

# a = [ rand(Float32,200) for i=1:25 ]
# k = map(KnetArray,a)
# @show vcat(a...) == vcat(k...)
# @show vcat(a...) == concat(k...)
# @show @benchmark vcat(a...)
# @show @benchmark vcat(k...)
# @show @benchmark concat(k...)
# @show @benchmark KnetArray(vcat(a...))
