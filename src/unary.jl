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

function unary_op(f, j=f, o...)
    J=Symbol(j)
    if isdefined(Base, J); eval(Expr(:import,:Base,J)); end
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
    @eval begin
        function $f{T<:AbstractFloat}(x::Array{T})
            y = similar(x)
            @inbounds for i=1:length(y)
                xi = x[i]
                y[i] = $y
            end
            return y
        end
        function $g{T<:AbstractFloat}(dy::Array{T},y::Array{T})
            dx = similar(dy)
            @inbounds for i=1:length(dx)
                yi = y[i]
                dyi = dy[i]
                dx[i] = $dx
            end
            return dx
        end
        $f{T<:Number}(xi::T)=$y
        $g{T<:Number}(dyi::T,yi::T)=$dx
        @primitive $f(x),dy,y $g(dy,y)
    end
end

"`invx(x) = (1./x)`" invx
"`relu(x) = max(0,x)`" relu
"`sigm(x) = (1./(1+exp(-x)))`" sigm

# To avoid conflict with AutoGrad:
@primitive tanh(x::Array),dy,y     tanhback(dy,y)
@primitive tanh(x::KnetArray),dy,y tanhback(dy,y)
@primitive tanhback(dy,y),ddx  ddx.*(1.-y.*y)  ddx.*(-2.*dy.*y)


"""

    logp(x,[dims])

Treat entries in `x` as as unnormalized log probabilities and return
normalized log probabilities.

`dims` is an optional argument, if not specified the normalization is
over the whole `x`, otherwise the normalization is performed over the
given dimensions.  In particular, if `x` is a matrix, `dims=1`
normalizes columns of `x` and `dims=2` normalizes rows of `x`.

"""

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

function logp(x,d...)
    if isa(x,Number)
        return zero(x)
    elseif isempty(x)
        return x
    else
        # x = x .- maximum(x,d...)
        # return (x .- log(sum(exp(x),d...)))
        # Expanding for profiling:
        x1 = maximum(x,d...)
        x2 = x .- x1
        x3 = exp(x2)
        x4 = sum(x3,d...)
        x5 = log(x4)
        x6 = x2 .- x5
        return x6
    end
end

function logpback(x,y,dy,d...)
    x = AutoGrad.getval(x)
    if isa(x,Number)
        return zero(x)
    elseif isempty(x)
        return x
    else
        # return (dy - exp(y).*sum(dy,d...))
        # Expanding for profiling:
        dx1 = sum(dy,d...)
        dx2 = exp(y)
        dx3 = dx2 .* dx1
        dx4 = dy - dx3
        return dx4
    end
end

# dy should be -p and y=logq so this should give us -p+q
@primitive  logp(x,d...),dy,y  logpback(x,y,dy,d...)


#=

# Dropout: work in progress

function dropout(x,p)
    if p > 0
        x .* (rand!(similar(x)) .> p) ./ (1-p)
    else
        x
    end
end

function dropback(x,p,y,dy)
    if x===y
        dy
    else
        dy .* (y .!== 0) ./ (1-p)  # TODO: test this
    end
end

@primitive  dropout(x,p),dy,y  dropback(x,p,y,dy)

# TODO: write efficient gpu kernel

=#
