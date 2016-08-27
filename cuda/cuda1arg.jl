using CUDArt
importall Base

cuda1arg = [
"sqrt",
# "rsqrt",
# "cbrt",
# "rcbrt",
"exp",
# "exp2",
# "exp10",
# "expm1",
"log",
# "log2",
# "log10",
# "log1p",
"sin",
# "cos",
# "tan",
# "sinpi",
# "cospi",
# "asin",
# "acos",
# "atan",
# "sinh",
# "cosh",
"tanh",
# "asinh",
# "acosh",
# "atanh",
# "erf",
# "erfc",
# "erfinv",
# "erfcinv",
# "erfcx",
# "normcdf",
# "normcdfinv",
# "lgamma",
# "tgamma",
# "logb",
# "ilogb",
# "j0",
# "j1",
# "y0",
# "y1",
# "cyl_bessel_i0",
# "cyl_bessel_i1",
# "trunc",
# "round",
# "rint",
# "nearbyint",
# "ceil",
# "floor",
# "lrint",
# "lround",
# "llrint",
# "llround",
("neg", "-", "-x[i]"),
("inv", "inv", "1/x[i]"),
("relu", "relu", "(x[i]>0?x[i]:0)"),
("sigm", "sigm", "1/(1+exp(-x[i]))"),
]

inv(x)=1./x
relu(x)=max(0,x)
sigm(x)=inv(1+exp(-x))

function cuda1def(f, j=f, o...)
    libknet1 = Pkg.dir("Knet/src/cuda/libknet1")
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$S"
        @eval begin
            function $J(x::CudaArray{$T})
                y = similar(x)
                ccall(($F,$libknet1),Void,(Cint,Ptr{$T},Ptr{$T}),length(y),x,y)
                return y
            end
        end
    end
end

for f in cuda1arg
    isa(f,Tuple) || (f=(f,))
    cuda1def(f...)
end
