#include "kunet.h"

__global__ void _At_mul_B_32(int nx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float *k) {
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
    k[i] = ki;
    i += blockDim.x * gridDim.x;
  }
}

// assume x(mx,nd) and s(nd,ns) are in 1-based csc format
// assume k(mx,ns) is a dense matrix that has been allocated

__global__ void _klinear32(int mx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float *k) {
  int s0, s1, sp, sc, sr, x0, x1, xp, xc, xr, k0, k1, kp;
  float sv, xv;
  sc = threadIdx.x + blockIdx.x * blockDim.x;
  while (sc < ns) {	// sc: 0-based column for s
    k0 = mx*sc;		// k[k0]: first element of k[:,sc]
    k1 = k0+mx;		// k[k1-1]: last element of k[:,sc]
    for (kp = k0; kp < k1; kp++) k[kp] = 0;
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
	k[k0+xr] += xv*sv;
      }
    }
    sc += blockDim.x * gridDim.x;
  }
}

__global__ void _klinear64(int mx, int ns, double *xval, int *xrow, int *xcol, double *sval, int *srow, int *scol, double *k) {
  int s0, s1, sp, sc, sr, x0, x1, xp, xc, xr, k0, k1, kp;
  double sv, xv;
  sc = threadIdx.x + blockIdx.x * blockDim.x;
  while (sc < ns) {	// sc: 0-based column for s
    k0 = mx*sc;		// k[k0]: first element of k[:,sc]
    k1 = k0+mx;		// k[k1-1]: last element of k[:,sc]
    for (kp = k0; kp < k1; kp++) k[kp] = 0;
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
	k[k0+xr] += xv*sv;
      }
    }
    sc += blockDim.x * gridDim.x;
  }
}

__global__ void _kpoly32(int n, float c, float d, float *k) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    k[i] = pow(k[i] + c, d);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void _kpoly64(int n, double c, double d, double *k) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    k[i] = pow(k[i] + c, d);
    i += blockDim.x * gridDim.x;
  }
}

extern "C" {

void At_mul_B_32(int nx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float *k) KCALL(_At_mul_B_32,nx,ns,xval,xrow,xcol,sval,srow,scol,k)
void klinear32(int mx, int ns, float *xval, int *xrow, int *xcol, float *sval, int *srow, int *scol, float *k) KCALL(_klinear32,mx,ns,xval,xrow,xcol,sval,srow,scol,k)
void klinear64(int mx, int ns, double *xval, int *xrow, int *xcol, double *sval, int *srow, int *scol, double *k) KCALL(_klinear64,mx,ns,xval,xrow,xcol,sval,srow,scol,k)
void kpoly32(int n, float c, float d, float *k) KCALL(_kpoly32,n,c,d,k)
void kpoly64(int n, double c, double d, double *k) KCALL(_kpoly64,n,c,d,k)

}


/*
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

function kback(l::KPerceptron, z::CudaArray{Float32})
    nptr = Cint[l.dn]; uptr = Cfloat[l.du]
    ccall((:kback32,libkunet), Void,
          (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat}),
          z,l.y,l.dw0,l.dw1,l.dj,nptr,uptr)
    l.dn = nptr[1]; l.du = uptr[1]
end

function kback(l::KPerceptron, z::CudaArray{Float64})
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

*/
