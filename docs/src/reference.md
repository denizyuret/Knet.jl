Reference
=========

Standard Library
----------------

Julia standard library functions supported by KnetArrays.

Function reference
------------------

We implement machine learning models in Knet using regular Julia code
and the `grad` function. Knet defines a few more utility functions
listed below. See `@doc <function>` for full details.

|function|description|
|:-------|:----------|
|`grad`|returns the gradient function.|
|`KnetArray`|constructs a GPU array.|
|`gradcheck`|compares gradients with numeric approximations.|
|`Knet.dir`|returns a path relative to Knet root.|
|`gpu`|determines which GPU Knet uses.|
|`relu`|returns `max(0,x)`|
|`sigm`|returns `(1./(1+exp(-x)))`|
|`invx`|returns `(1./x)`|
|`logp`|returns `x .- log(sum(exp(x),[dims]))`|
|`logsumexp`|returns `log(sum(exp(x),[dims]))`|
|`conv4`|executes convolutions or cross-correlations.|
|`pool`|replaces several adjacent values with their mean or maximum.|
|`mat`|reshapes its input into a two-dimensional matrix.|
|`update!`|updates the weight depending on the gradient and the parameters of the optimization method|


<!---
TODO: move this to reference.md, add @ref to fns, add new fns.
-->

Optimization methods
--------------------

In the examples above, we used simple SGD as the optimization method and
performed parameter updates manually using `w[i] -= lr * dw[i]`. The
`update!` function provides more optimization methods and can be used in
place of this manual update. In addition to a weight array `w[i]` and
its gradient `dw[i]`, `update!` requires a third argument encapsulating
the type, options, and state of the optimization method. The
constructors of the supported optimization methods are listed below. See
`@doc Sgd` etc. for full details. Note that in general we need to keep
one of these state variables per weight array, see
[optimizers.jl](https://github.com/denizyuret/Knet.jl/blob/master/examples/optimizers.jl)
for example usage.

|optimizer|parameters|
|:--------|:---------|
|`Sgd`|learning rate|
|`Momentum`|learning rate, gamma and velocity|
|`Adam`|learning rate, beta1, beta2, epsilon, time, first and second moments|
|`Adagrad`|learning rate, epsilon and accumulated gradients (G)|
|`Adadelta`|learning rate, rho, epsilon, accumulated gradients (G) and updates (delta)|
|`Rmsprop`|learning rate, rho, epsilon and accumulated gradients (G)|


Standard Library
----------------

Julia standard library functions supported by KnetArrays.

.. table:: Unary operations.

  +-------------+
  | -,  	|
  | abs,	|
  | abs2,	|
  | acos,	|
  | acosh,	|
  | asin,	|
  | asinh,	|
  | atan,	|
  | atanh,	|
  | cbrt,	|
  | ceil,	|
  | cos,	|
  | cosh,	|
  | cospi,	|
  | erf,	|
  | erfc,	|
  | erfcinv,	|
  | erfcx,	|
  | erfinv,	|
  | exp,	|
  | exp10,	|
  | exp2,	|
  | expm1,	|
  | floor,	|
  | log,	|
  | log10,	|
  | log1p,	|
  | log2,	|
  | round,	|
  | sign,	|
  | sin,	|
  | sinh,	|
  | sinpi,	|
  | sqrt,	|
  | tan,	|
  | tanh,	|
  | trunc,	|
  +-------------+


.. TODO link these functions to their Julia docs

..
   # Currently unsupported unary functions defined by cuda:
   # "cyl_bessel_i0",
   # "cyl_bessel_i1",
   # "ilogb",
   # "j0",
   # "j1",
   # "lgamma", # missing digamma for derivative
   # "llrint",
   # "llround",
   # "logb",
   # "lrint",
   # "lround",
   # "nearbyint",
   # "normcdf",
   # "normcdfinv",
   # "rcbrt",
   # "rint",
   # "rsqrt",
   # "tgamma",
   # "y0",
   # "y1",


..
   Unary functions not in standard library
   ("invx", "invx", "1/xi"),
   ("sigm", "sigm", "(xi>=0?1/(1+exp(-xi)):(exp(xi)/(1+exp(xi))))"),
   ("relu", "relu", "(xi>0?xi:0)"),
