# Gamma family

function cuda1gammafamily()
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
log_str = (T == "float") ? "logf" : "log"
max_str = (T == "float") ? "FLT_MAX" : "DBL_MAX"
pow_str = (T == "float") ? "powf" : "pow"
fabs_str = (T == "float") ? "fabsf" : "fabs"
one_str = (T == "float") ? "1.0f" : "1.0"
zeta_impl_series_2nd_cond = (T == "float") ? "" : "||(*a <= 9.0)"

print(s,
"""
__device__ __host__ $T digamma_impl_$F(const $T u) {

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

  yi = $log_str(s) - (half / s) - yi - w;

  return (negative) ? yi - nz : yi;

}

__device__ __host__ int zeta_impl_series_$F($T *a, $T *b, $T *s, const $T x,
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

__device__ __host__ $T zeta_impl_$F($T x, $T q) {
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
  if (zeta_impl_series_$F(&a, &b, &s, x, machep)) {
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

__device__ __host__ $T gamma_impl_$F($T x) {
  return tgamma(x);
}

__device__ __host__ $T polygamma_impl_$F(int n, $T x) {
  if (n == 0) {
    return digamma_impl_$F(x);
  }

  // dumb code to calculate factorials
  $T factorial = 1.0;
  for (int i = 0; i < n; i++) {
    factorial *= (i + 1);
  }

  return $pow_str(-1.0, n + 1) * factorial * zeta_impl_$F(n + 1, x);
}

__device__ __host__ $T trigamma_impl_$F($T x) {
  return polygamma_impl_$F(1, x);
}
""")
        end
    end
end
