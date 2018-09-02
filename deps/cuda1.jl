# Kernels for unary array operations

fp = open("cuda1.cu","w")
#using Knet: unary_ops


# Digamma

function cuda1gammafamily(; BLK=256, THR=256)
    sprint() do s
        print(s,
"""
#include <float.h>
#include <math.h>

__device__ __host__ float polynomial_evaluation_32(float x, const float *f, int n) {
  float result = 0.0;
  for (int i = 0; i < n; i++) {
    result *= x;
    result += f[i];
  }
  return result;
}

__device__ __host__ double polynomial_evaluation_64(double x, const double *f, int n) {
  double result = 0.0;
  for (int i = 0; i < n; i++) {
    result *= x;
    result += f[i];
  }
  return result;
}

__device__ __host__ float digamma_impl_maybe_poly_32(const float s) {
  const float A[] = {-4.16666666666666666667E-3f, 3.96825396825396825397E-3f,
                     -8.33333333333333333333E-3f, 8.33333333333333333333E-2f};
  float z;
  if (s < 1.0e8f) {
    z = 1.0f / (s * s);
    return z * polynomial_evaluation_32(z, A, 4);
  } else {
    return 0.0f;
  }
}

__device__ __host__ double digamma_impl_maybe_poly_64(const double s) {
  const double A[] = {8.33333333333333333333E-2, -2.10927960927960927961E-2,
                      7.57575757575757575758E-3, -4.16666666666666666667E-3,
                      3.96825396825396825397E-3, -8.33333333333333333333E-3,
                      8.33333333333333333333E-2};

  double z;
  if (s < 1.0e17) {
    z = 1.0 / (s * s);
    return z * polynomial_evaluation_64(z, A, 7);
  } else
    return 0.0;
}
""")
        for (T,F) in [("float","32"),("double","64")]
            floor_str = (T == "float") ? "floorf" : "floor"
            tan_str = (T == "float") ? "tanf" : "tan"
            max_str = (T == "float") ? "FLT_MAX" : "DBL_MAX"
            pow_str = (T == "float") ? "powf" : "pow"
            fabs_str = (T == "float") ? "fabsf" : "fabs"
            one_str = (T == "float") ? "1.0f" : "1.0"
            zeta_impl_series_2nd_cond = (T == "float") ? "" : "||(*a <= 9.0)"
            
            print(s,
"""
__device__ __host__ $T digamma_impl(const $T u) {

  $T xi = u;
  $T p, q, nz, s, w, yi;
  bool negative;

  const $T maxnum = $max_str;
  const $T m_pi = M_PI;

  negative = 0;
  nz = 0.0;

  const $T zero = 0.0;
  const $T one = 1.0;
  const $T half = 0.5;

  if (xi <= zero) {
    negative = one;
    q = xi;
    p = $floor_str(q);
    if (p == q) {
      return maxnum;
    }
    /* Remove the zeros of tan(m_pi x)
    * by subtracting the nearest integer from x
    */
    nz = q - p;
    if (nz != half) {
      if (nz > half) {
        p += one;
        nz = q - p;
      }
      nz = m_pi / $tan_str(m_pi * nz);
    } else {
      nz = zero;
    }
    xi = one - xi;
  }

  /* use the recurrence psi(x+1) = psi(x) + 1/x. */
  s = xi;
  w = zero;
  while (s < 10.0) {
    w += one / s;
    s += one;
  }

  yi = digamma_impl_maybe_poly_$F(s);

  yi = logf(s) - (half / s) - yi - w;

  return (negative) ? yi - nz : yi;

}

__device__ __host__ int zeta_impl_series($T *a, $T *b, $T *s, const $T x,
                                         const $T machep) {
  int i = 0;
  while ((i < 9)$zeta_impl_series_2nd_cond) {
    i += 1;
    *a += $one_str;
    *b = $pow_str(*a, -x);
    *s += *b;
    if ($fabs_str(*b / *s) < machep) {
      return true;
    }
  }

  // Return whether we are done
  return false;
}

__device__ __host__ $T zeta_impl($T x, $T q) {
  int i;
  $T p, r, a, b, k, s, t, w;

  const $T A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12,  /*1.067062284288e16/3617*/
      1.1646782814350067249e14,  /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17,  /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091*/
  };

  const $T maxnum = $max_str;
  const $T zero = 0.0, half = 0.5, one = 1.0;
  const $T machep = 1e-15;

  if (x == one) return maxnum;

  if (x < one) {
    return zero;
  }

  if (q <= zero) {
    if (q == $floor_str(q)) {
      return maxnum;
    }
    p = x;
    r = $floor_str(p);
    if (p != r) return zero;
  }

  /* Permit negative q but continue sum until n+q > +9 .
   * This case should be handled by a reflection formula.
   * If q<0 and x is an integer, there is a relation to
   * the polygamma function.
   */
  s = $pow_str(q, -x);
  a = q;
  b = zero;

  // Run the summation in a helper function that is specific to the floating
  // precision
  if (zeta_impl_series(&a, &b, &s, x, machep)) {
    return s;
  }

  w = a;
  s += b * w / (x - one);
  s -= half * b;
  a = one;
  k = zero;
  for (i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = $fabs_str(t / s);
    if (t < machep) return s;
    k += one;
    a *= x + k;
    b /= w;
    k += one;
  }
  return s;
};

__device__ __host__ $T polygamma_impl(int n, $T x) {
  if (n == 0) {
    return digamma_impl(x);
  }

  // dumb code to calculate factorials
  $T factorial = 1.0;
  for (int i = 0; i < n; i++) {
    factorial *= (i + 1);
  }

  return $pow_str(-1.0, n + 1) * factorial * zeta_impl(n + 1, x);
}

__device__ __host__ $T gamma_impl($T x) {
  return exp(lgamma(x));
}

__device__ __host__ $T trigamma_impl($T x) {
  return polygamma_impl(1, x);
}
""")
        end
    end
end

print(fp,cuda1gammafamily())

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
  $DLLEXPORT void $F(int n, $T *x, $T *y) {
    if (n>0) _$F<<<$BLK,$THR>>>(n,x,y);
  }    
}
""")
        end
    end
end

for a in unary_ops
    if !isa(a,Tuple); a=(a,); end
    print(fp,cuda1src(a...))
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
  $DLLEXPORT void fill_$F(int n, $T x, $T *y) {
    if (n>0) _fill_$F<<<$BLK,$THR>>>(n,x,y);
  }    
}
""")
        end
    end
end

print(fp,cuda1fill())

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
  $DLLEXPORT void xfill_$F(int nrows, int ncols, $T x, $T *y, int incy) {
    if (nrows>0 && ncols>0) _xfill_$F<<<$BLK,$THR>>>(nrows, ncols, x, y, incy);
  }    
}
""")
        end
    end
end

print(fp,cuda1xfill())

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
  $DLLEXPORT void xcopy(int nrows, int ncols, const void *x, int incx, void *y, int incy) {
    if (nrows>0 && ncols>0) _xcopy<<<$BLK,$THR>>>(nrows,ncols,(char*)x,incx,(char*)y,incy);
  }    
}
"""
end

print(fp,cuda1xcopy())


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
  $DLLEXPORT void $(F)($T* x, int dimx1, int dimx2, $T* y, int dimy1) {
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
  $DLLEXPORT void $(F)($T* x, int dimx1, int dimx2, int dimx3, $T* y, int dimy1, int dimy2) {
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
  $DLLEXPORT void $(F)($T* x, int dimx1, int dimx2, int dimx3, int dimx4, $T* y, int dimy1, int dimy2, int dimy3) {
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
  $DLLEXPORT void $(F)($T* x, int dimx1, int dimx2, int dimx3, int dimx4, int dimx5, $T* y, int dimy1, int dimy2, int dimy3, int dimy4) {
    _$(F)<<<$BLK,$THR>>>(x,dimx1,dimx2,dimx3,dimx4,dimx5,y,dimy1,dimy2,dimy3,dimy4);
  }    
}
""")
        end
    end
end

# avoid dependency, copy permutations from julia4 base
# using Combinatorics

import Base: iterate, length
struct Perms; a; end
perms(a) = Perms(a)
length(p::Perms) = factorial(length(p.a))
#start(p::Perms) = [1:length(p.a);]
#done(p::Perms, s) = !isempty(s) && s[1] > length(p.a)
function iterate(p::Perms, s=[1:length(p.a);])
    if !isempty(s) && s[1] > length(p.a); return nothing; end
    perm = [p.a[si] for si in s]
    if length(p.a) == 0
        # special case to generate 1 result for len==0
        return (perm,[1])
    end
    s = copy(s)
    k = length(s)-1
    while k > 0 && s[k] > s[k+1];  k -= 1;  end
    if k == 0
        s[1] = length(s)+1   # done
    else
        l = length(s)
        while s[k] >= s[l];  l -= 1;  end
        s[k],s[l] = s[l],s[k]
        reverse!(s,k+1)
    end
    (perm,s)
end

function cuda1permutedims()
  cudaPerms = [permutedims2Dsrc,permutedims3Dsrc,permutedims4Dsrc,permutedims5Dsrc]
  for i=2:5
      dims = collect(perms(1:i))
      indnames = collect(perms(["i","j","k","l","m"][1:i]))
      for j=1:length(dims)
          fname = string("permutedims_",i,"D_",join(dims[j],"_"),"_")
          print(fp,cudaPerms[i-1](fname,indnames[j]...))
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
  $DLLEXPORT void icat_$F(int nrows, int ncols, $T **x, $T *y) {
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

print(fp,cuda1icat())

# This is for missing double atomicAdd()
print(fp,"""
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
$DLLEXPORT void getcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y)
{ if (ncols>0 && xrows>0 && xcols>0) _getcols_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
$DLLEXPORT void setcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y)
{ if (ncols>0 && xrows>0 && xcols>0) _setcols_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
$DLLEXPORT void addcols_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T *y)
{ if (ncols>0 && xrows>0 && xcols>0) _addcols_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
$DLLEXPORT void setcol1_$F(int xrows, int xcols, int ncols, int *cols, $T *x, $T  y)
{ if (ncols>0 && xrows>0 && xcols>0) _setcol1_$F<<<$BLK,$THR>>>(xrows,xcols,ncols,cols,x,y); }
$DLLEXPORT void getrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y)
{ if (nrows>0 && xrows>0 && xcols>0) _getrows_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
$DLLEXPORT void setrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y)
{ if (nrows>0 && xrows>0 && xcols>0) _setrows_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
$DLLEXPORT void addrows_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T *y)
{ if (nrows>0 && xrows>0 && xcols>0) _addrows_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
$DLLEXPORT void setrow1_$F(int xrows, int xcols, int nrows, int *rows, $T *x, $T  y)
{ if (nrows>0 && xrows>0 && xcols>0) _setrow1_$F<<<$BLK,$THR>>>(xrows,xcols,nrows,rows,x,y); }
$DLLEXPORT void getents_$F(int n, int *ents, $T *x, $T *y)
{ if (n>0) _getents_$F<<<$BLK,$THR>>>(n,ents,x,y); }
$DLLEXPORT void setents_$F(int n, int *ents, $T *x, $T *y)
{ if (n>0) _setents_$F<<<$BLK,$THR>>>(n,ents,x,y); }
$DLLEXPORT void addents_$F(int n, int *ents, $T *x, $T *y)
{ if (n>0) _addents_$F<<<$BLK,$THR>>>(n,ents,x,y); }
$DLLEXPORT void setent1_$F(int n, int *ents, $T *x, $T  y)
{ if (n>0) _setent1_$F<<<$BLK,$THR>>>(n,ents,x,y); }
}
""")
        end
    end
end

print(fp,cuda1getcols())


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
  $DLLEXPORT void dropout_$F(int n, $T p, $T *x, $T *y) {
    if (n>0) _dropout_$F<<<$BLK,$THR>>>(n,p,1.0/(1.0-p),x,y);
  }    
  $DLLEXPORT void dropback_$F(int n, $T p, $T *x, $T *y, $T *dy, $T *dx) {
    if (n>0) _dropback_$F<<<$BLK,$THR>>>(n,1.0/(1.0-p),y,dy,dx);
  }    
}
""")
        end
    end
end

print(fp,cuda1dropout())

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
  $DLLEXPORT void concat_$F(int narrays, int *starts, int *lengths, $T **x, $T *y) {
    _concat_$F<<<narrays,$THR>>>(narrays, starts, lengths, x, y);
  }    
}
""")
        end
    end
end

print(fp,cuda1concat())

close(fp)

# Here is the test script for cuda1concat:
# using Knet, BenchmarkTools
# using Knet: @knet8

# for S in (32,64); T = Symbol("Float$S"); F = "concat_$S"
# @eval function concat(A::KnetArray{$T}...)
#     nargs = length(A)
#     S = Array{Int32}(nargs)
#     L = Array{Int32}(nargs)
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
