# The entry format is (cudaname, julianame, kernelcode)
# With single name entries cudaname=julianame and kernelcode=name(xi,yi).
# I commented out functions if I don't know the Julia equivalent.

binary_ops = [
    ("add","+","xi+yi"),
    ("sub","-","xi-yi"),
    ("mul","*","xi*yi"),
    ("div","/","xi/yi"),
    ("pow","^","pow(xi,yi)"),
    ("max","max","(xi>yi?xi:yi)"),
    ("min","min","(xi<yi?xi:yi)"),
    ("eq","==","xi==yi"),
    ("ne","!=","xi!=yi"),
    ("gt",">","xi>yi"),
    ("ge",">=","xi>=yi"),
    ("lt","<","xi<yi"),
    ("le","<=","xi<=yi"),
    # "hypot",
    # "rhypot",
    # "atan2",
    # "frexp",
    # "ldexp",
    # "scalbn",
    # "scalbln",
    # "fmod",
    # "remainder",
    # "mod",
    # "fdim",
    #("invxback","invxback","(-xi*yi*yi)"),
    ("reluback","reluback","(yi>0?xi:0)"),
    ("eluback","eluback","(yi>0?xi:xi*(1+yi))"),
    ("seluback","seluback","(yi>0?1.0507009873554805*xi:xi*(1.7580993408473773+yi))"),
    ("sigmback","sigmback","(xi*yi*(1-yi))"),
    ("tanhback","tanhback","(xi*(1-yi*yi))"),
    # ("rpow","rpow","pow(yi,xi)"),   # need this for Array.^Scalar -> cuda bug #108 switching to CUDA.jl for pow
]

unary_ops_with_int_degree = [
    # cuda does not define negative degrees for bessel, fix it here:
    ("jn", "besselj", "(d>=0 ? jn(d,xi) : d%2==0 ? jn(-d,xi) : -jn(-d,xi))"),
    ("yn", "bessely", "(d>=0 ? yn(d,xi) : d%2==0 ? yn(-d,xi) : -yn(-d,xi))"), # bessely not defined for negative x!
]

# The following list comes from the NVIDIA math docs with some extras.
# http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix
# The entry format is (cudaname, julianame, kernelcode)
# With single name entries cudaname=julianame and kernelcode=name(xi).
# I commented out functions if I don't know the Julia equivalent.
unary_ops = [
("abs2", "abs2", "(xi*xi)"),
("abs", "abs", "(xi<0?-xi:xi)"),
"acos",
"acosh",
"asin",
"asinh",
"atan",
"atanh",
"cbrt",
"ceil",
"cos",
"cosh",
"cospi",
# ("cyl_bessel_i0", "besseli0"),  # besseli0,i1 is not defined in SpecialFunctions
# ("cyl_bessel_i1", "besseli1"),
"erf",
"erfc",
"erfcinv",
"erfcx",
"erfinv",
"exp",
"exp10",
"exp2",
"expm1",
"floor",
# "ilogb",
#("invx", "invx", "1/xi"),
("j0", "besselj0"),
("j1", "besselj1"),
("gamma_impl", "gamma"),
("lgamma", "lgamma"), # TODO: SpecialFunctions 0.8: lgamma(x::Real)` is deprecated, use `(logabsgamma(x))[1]` instead. Other alternative is loggamma, throws a DomainError if gamma(x) is negative.
("digamma_impl", "digamma"),
("trigamma_impl", "trigamma"),
#
# "llrint",
# "llround",
"log",
"log10",
"log1p",
"log2",
# "logb",
# "lrint",
# "lround",
# "nearbyint",
("neg", "-", "-xi"),
# "normcdf",
# "normcdfinv",
("one", "one", "1"),
# "rcbrt",
("relu", "relu", "(xi>0?xi:0)"),
("elu", "elu", "(xi>0?xi:exp(xi)-1)"),
("selu", "selu", "1.0507009873554805*(xi>0?xi:1.6732632423543778*(exp(xi)-1))"),
# "rint",
"round",
# "rsqrt",
("sigm", "sigm", "(xi>=0?1/(1+exp(-xi)):(exp(xi)/(1+exp(xi))))"),
("sign", "sign", "(xi>0?1:xi<0?-1:0)"),
"sin",
"sinh",
"sinpi",
"sqrt",
"tan",
"tanh",
# "tgamma",
"trunc",
("y0","bessely0"),
("y1","bessely1"),
("zero", "zero", "0"),
]

reduction_ops = [
# The entry format is (cudaname, julianame, merge, item, init)
# ai is the accumulator, xi is the array element
("sum","sum","ai+xi","xi","0"),
("prod","prod","ai*xi","xi","1"),
("maximum","maximum","(ai>xi ? ai : xi)","xi","(-INFINITY)"),
("minimum","minimum","(ai<xi ? ai : xi)","xi","INFINITY"),
("sumabs","sumabs","ai+xi","(xi<0 ? -xi : xi)","0"),
("sumabs2","sumabs2","ai+xi","(xi*xi)","0"),
("maxabs","maxabs","(ai>xi ? ai : xi)","(xi<0 ? -xi : xi)","0"),
("minabs","minabs","(ai<xi ? ai : xi)","(xi<0 ? -xi : xi)","INFINITY"),
("countnz","countnz","ai+xi","(xi!=0)","0"),
]
