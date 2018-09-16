# The entry format is (cudaname, julianame, kernelcode)
# With single name entries cudaname=julianame and kernelcode=name(xi,yi).
# I commented out functions if I don't know the Julia equivalent.

broadcast_ops = [
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
    # "jn",
    # "yn",
    # "fmod",
    # "remainder",
    # "mod",
    # "fdim",
    ("invxback","invxback","(-xi*yi*yi)"),
    ("reluback","reluback","(yi>0?xi:0)"),
    ("eluback","eluback","(yi>0?xi:xi*(1+yi))"),
    ("sigmback","sigmback","(xi*yi*(1-yi))"),
    ("tanhback","tanhback","(xi*(1-yi*yi))"),
    ("rpow","rpow","pow(yi,xi)"),   # need this for Array.^Scalar
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
# "cyl_bessel_i0",
# "cyl_bessel_i1",
"exp",
"exp10",
"exp2",
"expm1",
"floor",
# "ilogb",
("invx", "invx", "1/xi"),
# "j0",
# "j1",
# "lgamma", # missing digamma for derivative
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
("elu", "elu", "(xi>0?xi:exp(xi)-1)"),
("relu", "relu", "(xi>0?xi:0)"),
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
# "y0",
# "y1",
("zero", "zero", "0"),
]

if true #TODO Pkg.installed("SpecialFunctions") != nothing
    append!(unary_ops, [
"erf",     # Removed from base in julia6
"erfc",
"erfcinv",
"erfcx",
"erfinv",
])
end

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

