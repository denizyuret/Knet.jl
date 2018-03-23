# unary.jl: Unary Array->Array operations.
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
# "rcbrt",
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
]

using Compat.Pkg
if installed("SpecialFunctions") != nothing
    append!(unary_ops, [
"erf",     # Removed from base in julia6
"erfc",
"erfcinv",
"erfcx",
"erfinv",
])
end

function unary_op(f, j=f, o...)
    J=broadcast_func(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$S"
        @eval begin
            function $J(x::KnetArray{$T})
                y = similar(x)
                @knet8($F,(Cint,Ptr{$T},Ptr{$T}),length(y),x,y)
                return y
            end
        end
    end
end

for f in unary_ops
    if !isa(f,Tuple); f=(f,); end
    unary_op(f...)
end

# Define some common operations as primitives for efficiency:
# 1. Avoid creating intermediate arrays
# 2. Avoid taking derivatives of intermediate operations

for (f,g,y,dx) in
    ((:invx, :invxback, :(one(T)/xi), :(-yi*yi*dyi)),
     (:relu, :reluback, :(max(zero(T),xi)), :(ifelse(yi>0,dyi,zero(T)))),
     (:tanx, :tanhback, :(tanh(xi)), :(dyi*(one(T)-yi*yi))),
     (:sigm, :sigmback, 
      # Numerically stable implementation from
      # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
      :(if xi>=0; z=exp(-xi); one(T)/(one(T)+z); else; z=exp(xi); z/(one(T)+z); end),
      :(dyi*yi*(one(T)-yi))),
     )
    bf = broadcast_func(f)
    bg = broadcast_func(g)
    @eval begin
        function $bf(x::Array{T}) where {T<:AbstractFloat}
            y = similar(x)
            @inbounds for i=1:length(y)
                xi = x[i]
                y[i] = $y
            end
            return y
        end
        function $bg(dy::Array{T},y::Array{T}) where {T<:AbstractFloat}
            dx = similar(dy)
            @inbounds for i=1:length(dx)
                yi = y[i]
                dyi = dy[i]
                dx[i] = $dx
            end
            return dx
        end
        $f(xi::T) where {T<:Number}=$y
        $g(dyi::T,yi::T) where {T<:Number}=$dx
        @primitive $f(x),dy,y $g(dy,y)
        if $f != $bf
            @primitive $bf(x),dy,y $bg(dy,y)
        end
    end
end

"`invx(x) = (1./x)`" invx
"`relu(x) = max(0,x)`" relu
"`sigm(x) = (1./(1+exp(-x)))`" sigm

# To avoid conflict with AutoGrad:
# TODO: test this in Julia6, do we need to fix broadcast_func(tanh)?
import Base: tanh
@primitive tanh(x::Array),dy,y     tanhback(dy,y)
@primitive tanh(x::KnetArray),dy,y tanhback(dy,y)
@primitive tanhback(dy,y),ddx  ddx .* (1 .- y .* y)  ddx .* (-2 .* dy .* y)

# Unary plus and minus
import Base: +, .+, -, .-, broadcast

broadcast(::typeof(+), a::KnetArray)=a
+(a::KnetArray)=a
-(a::KnetArray)=broadcast(-,a)

