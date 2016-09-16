import Base: sqrt, exp, log, sin, tanh, -, abs, abs2

cuda1 = [
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
("neg", "-", "-xi"),
("invx", "invx", "1/xi"),
("relu", "relu", "(xi>0?xi:0)"),
("sigm", "sigm", "1/(1+exp(-xi))"),
("abs", "abs", "(xi<0?-xi:xi)"),
("abs2", "abs2", "(xi*xi)"),
]

function cuda1def(f, j=f, o...)
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$S"
        @eval begin
            function $J(x::KnetArray{$T})
                y = similar(x)
                ccall(($F,$libknet8),Void,(Cint,Ptr{$T},Ptr{$T}),length(y),x,y)
                return y
            end
        end
    end
end

#if isdefined(:libknet8)
    for f in cuda1
        isa(f,Tuple) || (f=(f,))
        cuda1def(f...)
    end
#end

# Define some common operations as primitives for efficiency:
# 1. Avoid creating intermediate arrays
# 2. Avoid taking derivatives of intermediate operations

for (f,g,y,dx) in ((:invx, :invxback, :(one(T)/x[i]), :(-y[i]*y[i]*dy[i])),
                   (:relu, :reluback, :(max(zero(T),x[i])), :(ifelse(y[i]>0,dy[i],zero(T)))),
                   (:sigm, :sigmback, :(one(T)/(one(T)+exp(-x[i]))), :(dy[i]*y[i]*(one(T)-y[i]))),
                   (:tanx, :tanhback, :(tanh(x[i])), :(dy[i]*(one(T)-y[i]*y[i]))),
                   )
    @eval begin
        function $f{T<:AbstractFloat}(x::Array{T})
            y = similar(x)
            @inbounds for i=1:length(y)
                y[i] = $y
            end
            return y
        end
        function $g{T<:AbstractFloat}(dy::Array{T},y::Array{T})
            dx = similar(dy)
            @inbounds for i=1:length(dx)
                dx[i] = $dx
            end
            return dx
        end
        @primitive $f(x),dy,y $g(dy,y)
    end
end

# To avoid conflict with AutoGrad:
@primitive tanh(x::Array),dy,y     tanhback(dy,y)
@primitive tanh(x::KnetArray),dy,y tanhback(dy,y)


# Math for the cross-entropy loss: x is unnormalized input, p is
# target probabilities, q is estimated probabilities. Read left column
# down, right column (loss gradients) back up.

# x			dx = -p + qz/z = -p + exp(logq)
# xmax  = max(x,1)	-sum(db)=0
# logqz = x .- xmax	-p + qz/z
# qz    = exp(logqz)	rep(1/z)
# z     = sum(qz,1)	1/z
# logz  = log(z)	sum(p)=1
# logq  = logqz.-logz	-p
# plogq = p .* logq	-1
# loss  = -sum(plogq)	1

"Treat columns of x as unnormalized logp and return normalized logp."
function logp(x)
    x = x .- maximum(x,1)
    x = x .- log(sum(exp(x),1))
end

# dy should be -p and y=logq so this should give us -p+q
@primitive  logp(x),dy,y  (dy - exp(y).*sum(dy,1))
