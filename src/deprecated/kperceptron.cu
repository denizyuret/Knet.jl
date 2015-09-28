#include "kunet.h"

/* kgauss uses the same algorithm and input/output format as At_mul_B
   to compute the gaussian kernel: x(nd,nx) s(nd,ns) -> k(nx,ns) 
   Input: two sparse matrices.  Output: a dense matrix.
   The sparse matrices are in 1-based csc format.
*/

__global__ void _kgauss32(int nx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float *k, float g) {
  int i, n, x1, x2, xc, xr, s1, s2, sc, sr;
  double d, dd;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  n = nx*ns;
  while (i < n) {
    xc = i % nx;
    sc = i / nx;
    x1 = xcol[xc]-1; x2 = xcol[xc+1]-1;
    s1 = scol[sc]-1; s2 = scol[sc+1]-1;
    dd = 0;
    while ((x1 < x2) || (s1 < s2)) {
      xr = ((x1 < x2) ? xrow[x1] : INT_MAX);
      sr = ((s1 < s2) ? srow[s1] : INT_MAX);
      d = ((sr < xr) ? sval[s1++] :
	   (xr < sr) ? xval[x1++] :
	   (xval[x1++]-sval[s1++])); 
      dd += d*d;
    }
    k[i] = exp(-g * dd);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _kgauss64(int nx, int ns, double *xval, int *xrow, int *xcol, double *sval, int *srow, int *scol, double *k, double g) {
  int i, n, x1, x2, xc, xr, s1, s2, sc, sr;
  double d, dd;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  n = nx*ns;
  while (i < n) {
    xc = i % nx;
    sc = i / nx;
    x1 = xcol[xc]-1; x2 = xcol[xc+1]-1;
    s1 = scol[sc]-1; s2 = scol[sc+1]-1;
    dd = 0;
    while ((x1 < x2) || (s1 < s2)) {
      xr = ((x1 < x2) ? xrow[x1] : INT_MAX);
      sr = ((s1 < s2) ? srow[s1] : INT_MAX);
      d = ((sr < xr) ? sval[s1++] :
	   (xr < sr) ? xval[x1++] :
	   (xval[x1++]-sval[s1++])); 
      dd += d*d;
    }
    k[i] = exp(-g * dd);
    i += blockDim.x * gridDim.x;
  }
}

/* kpoly uses the same algorithm and input/output format as At_mul_B
   to compute the polynomial kernel: x(nd,nx) s(nd,ns) -> k(nx,ns) */

__global__ void _kpoly32(int nx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float *k, float c, float d) {
  int i, n, x1, x2, xc, xr, s1, s2, sc, sr;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  n = nx*ns;
  while (i < n) {
    double ki = 0;
    xc = i % nx;
    sc = i / nx;
    x1 = xcol[xc]-1; x2 = xcol[xc+1]-1;
    s1 = scol[sc]-1; s2 = scol[sc+1]-1;
    while ((x1 < x2) && (s1 < s2)) {
      xr = xrow[x1]; sr = srow[s1];
      if (sr < xr) s1++;
      else if (xr < sr) x1++;
      else ki += xval[x1++]*sval[s1++];
    }
    k[i] = pow(ki + c, (double) d);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _kpoly64(int nx, int ns, double *xval, int *xrow, int *xcol, double *sval, int *srow, int *scol, double *k, double c, double d) {
  int i, n, x1, x2, xc, xr, s1, s2, sc, sr;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  n = nx*ns;
  while (i < n) {
    double ki = 0;
    xc = i % nx;
    sc = i / nx;
    x1 = xcol[xc]-1; x2 = xcol[xc+1]-1;
    s1 = scol[sc]-1; s2 = scol[sc+1]-1;
    while ((x1 < x2) && (s1 < s2)) {
      xr = xrow[x1]; sr = srow[s1];
      if (sr < xr) s1++;
      else if (xr < sr) x1++;
      else ki += xval[x1++]*sval[s1++];
    }
    k[i] = pow(ki + c, d);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _kpolymap32(int n, float *k, float c, float d) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    k[i] = pow(k[i] + c, d);
    i += blockDim.x * gridDim.x;
  }  
}

__global__ void _kpolymap64(int n, double *k, double c, double d) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    k[i] = pow(k[i] + c, d);
    i += blockDim.x * gridDim.x;
  }  
}

__global__ void _kgauss32map(int nx, int ns, float *x2, float *s2, float *k, float g) {
  int i, n, xi, si;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  n = nx*ns;
  while (i < n) {
    xi = (i % nx);
    si = (i / nx);
    k[i] = exp(-g * (x2[xi] + s2[si] - 2*k[i]));
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _kgauss64map(int nx, int ns, double *x2, double *s2, double *k, double g) {
  int i, n, xi, si;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  n = nx*ns;
  while (i < n) {
    xi = (i % nx);
    si = (i / nx);
    k[i] = exp(-g * (x2[xi] + s2[si] - 2*k[i]));
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _kgauss32sum(int xrows, int xcols, float *x, float *xx) {
  int i, j, x0, x1;
  double sum;
  j = threadIdx.x + blockIdx.x * blockDim.x;
  while (j < xcols) {
    x0 = j*xrows; x1 = x0+xrows;
    sum = 0;
    for (i=x0; i<x1; i++) sum += x[i]*x[i];
    xx[j] = sum;
    j += blockDim.x * gridDim.x;
  }
}

__global__ void _kgauss64sum(int xrows, int xcols, double *x, double *xx) {
  int i, j, x0, x1;
  double sum;
  j = threadIdx.x + blockIdx.x * blockDim.x;
  while (j < xcols) {
    x0 = j*xrows; x1 = x0+xrows;
    sum = 0;
    for (i=x0; i<x1; i++) sum += x[i]*x[i];
    xx[j] = sum;
    j += blockDim.x * gridDim.x;
  }
}

extern "C" {

  void kgauss32map(int nx, int ns, float *x2, float *s2, float *k, float g) KCALL(_kgauss32map,nx,ns,x2,s2,k,g);
  void kgauss32sum(int xrows, int xcols, float *x, float *x2) KCALL(_kgauss32sum,xrows,xcols,x,x2);

  void kgauss64map(int nx, int ns, double *x2, double *s2, double *k, double g) KCALL(_kgauss64map,nx,ns,x2,s2,k,g);
  void kgauss64sum(int xrows, int xcols, double *x, double *x2) KCALL(_kgauss64sum,xrows,xcols,x,x2);

  void kpolymap32(int n, float *k, float c, float d) KCALL(_kpolymap32,n,k,c,d);
  void kpolymap64(int n, double *k, double c, double d) KCALL(_kpolymap64,n,k,c,d);

  void kpoly32(int nx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float *k, float c, float d) KCALL(_kpoly32,nx,ns,xval,xrow,xcol,sval,srow,scol,k,c,d);
  void kpoly64(int nx, int ns, double *xval, int *xrow, int *xcol, double *sval, int *srow, int *scol, double *k, double c, double d) KCALL(_kpoly64,nx,ns,xval,xrow,xcol,sval,srow,scol,k,c,d);

  void kgauss32(int nx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float *k, float g) KCALL(_kgauss32,nx,ns,xval,xrow,xcol,sval,srow,scol,k,g);
  void kgauss64(int nx, int ns, double *xval, int *xrow, int *xcol, double *sval, int *srow, int *scol, double *k, double g) KCALL(_kgauss64,nx,ns,xval,xrow,xcol,sval,srow,scol,k,g);

}


/* DEAD CODE...
// no need for kback on gpu?

__global__ void _kback32(int nc, int nx, float *z, float *y, float *dw0, float *dw1, int *dj, float u) {
  int i0, i1, cz, cy;
  float cmax, ymax;
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  while (ix < nx) {
    i0 = ix * nc; 
    i1 = i0 + nc;
    ymax = -INFINITY; cy = -1;
    zmax = -INFINITY; cz = -1;
    for (int i=i0; i<i1; i++) {
      if (y[i] > ymax) { ymax = y[i]; cy = i; }
      if (z[i] > zmax) { zmax = z[i]; cz = i; }
    }
    if (cy != cz) {
      
    }
    ix += blockDim.x * gridDim.x;
  }
}


__global__ void _drop32(int n, float *x, float *xmask, float dropout, float scale) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (xmask[i] < dropout) x[i] = 0;
    else x[i] *= scale;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _drop64(int n, double *x, double *xmask, double dropout, double scale) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    if (xmask[i] < dropout) x[i] = 0;
    else x[i] *= scale;
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {

void kback32(int nc, int nx, float *z, float *y, float *dw0, float *dw1, int *dj, int *dn, float *du) {

}

void drop32(int n, float *x, float *xmask, float dropout, float scale) KCALL(_drop32,n,x,xmask,dropout,scale);
void drop64(int n, double *x, double *xmask, double dropout, double scale) KCALL(_drop64,n,x,xmask,dropout,scale);



}

if GPU

function kback(l::KPerceptron, z::AbstractCudaArray{Float32})
    nptr = Cint[l.dn]; uptr = Cfloat[l.du]
    ccall((:kback32,libkunet), Void,
          (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          z,l.y,l.dw0,l.dw1,l.dj,nptr,uptr)
    l.dn = nptr[1]; l.du = uptr[1]
end

function kback(l::KPerceptron, z::AbstractCudaArray{Float64})
    nptr = Cint[l.dn]; uptr = Cdouble[l.du]
    ccall((:kback64,libkunet), Void,
          (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble}),
          z,l.y,l.dw0,l.dw1,l.dj,nptr,uptr)
    l.dn = nptr[1]; l.du = uptr[1]
end

end # if GPU

// buggy: does not process when one matrix has a zero 
__global__ void _kgauss32(int mx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float g, float *k) {
  // assume x(mx,nd) and s(nd,ns) are in 1-based csc format
  // assume k(mx,ns) has been allocated and zeroed out
  int s0, s1, sp, sc, sr, x0, x1, xp, xc, xr, k0, k1, kp;
  float sv, xv, xs;
  sc = threadIdx.x + blockIdx.x * blockDim.x;
  k0 = mx*sc;		// k[k0]: first element of k[:,sc]
  k1 = k0+mx;		// k[k1-1]: last element of k[:,sc]
  while (sc < ns) {	// sc: 0-based column for s
    s0 = scol[sc]-1;    // first element of s[:,sc] is at sval[s0] (scol entries are 1-based)
    s1 = scol[sc+1]-1;  // last element of s[:,sc] is at sval[s1-1]
    for (sp = s0; sp < s1; sp++) {
      sr = srow[sp]-1;  // sr: 0-based row for s (srow entries are 1-based)
      sv = sval[sp];	// sv: s[sr,sc] (0-based)
      xc = sr;		// xc: 0-based column for x (=sr)
      x0 = xcol[xc]-1;  // first element of x[:,xc] is at xval[x0]
      x1 = xcol[xc+1]-1; // last element of x[:,xc] is at xval[x1-1]
      for (xp = x0; xp < x1; xp++) {
	xr = xrow[xp]-1; // xr: 0-based row for x
	xv = xval[xp];	 // xv: x[xr,xc=sr], now we can set k[xr,sc]
	xs = xv - sv;
	k[k0+xr] += xs*xs; // k += (xi-si)^2
      }
    }
    for (kp = k0; kp < k1; kp++) {
      k[kp] = exp(-g*k[kp]); // k = exp(-g*sum((xi-si)^2))
    }
    sc += blockDim.x * gridDim.x;
  }
}

extern "C" {
  void kgauss32(int mx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float g, float *k) {
    float *x2 = calloc(mx, sizeof(float));
    float *s2 = calloc(ns, sizeof(float));
    KCALL(_rowsq,mx,xval,xrow,xcol,x2);
    KCALL(_colsq,ns,sval,srow,scol,s2);
    KCALL(_kgauss32,mx,ns,xval,xrow,xcol,sval,srow,scol,g,k)
  }
}

__global__ void _kgauss32d(int nx, int ns, int nd, float *x, float *s, float *k, float g) {
  int i, j, n, xj, sj;
  double d, dd;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  n = nx*ns;
  while (i < n) {
    xj = (i % nx)*nd;
    sj = (i / nx)*nd;
    dd = 0;
    for (j = 0; j < nd; j++) {
      d = x[xj++]-s[sj++];
      dd += d*d;
    }
    k[i] = exp(-g * dd);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _kgauss64d(int nx, int ns, int nd, double *x, double *s, double *k, double g) {
  int i, j, n, xj, sj;
  double d, dd;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  n = nx*ns;
  while (i < n) {
    xj = (i % nx)*nd;
    sj = (i / nx)*nd;
    dd = 0;
    for (j = 0; j < nd; j++) {
      d = x[xj++]-s[sj++];
      dd += d*d;
    }
    k[i] = exp(-g * dd);
    i += blockDim.x * gridDim.x;
  }
}

  void kgauss32d(int nx, int ns, int nd, float *x, float *s, float *k, float g) KCALL(_kgauss32d,nx,ns,nd,x,s,k,g);
  void kgauss64d(int nx, int ns, int nd, double *x, double *s, double *k, double g) KCALL(_kgauss64d,nx,ns,nd,x,s,k,g);

*/
