# cuda10: Scalar,Array->Array; Array,Scalar->Array
# Most operations are symmetric, but for others:
# Kernels implement (s,a) order by default.
# So we can use a-s=-s+a and a/s=(1/s)*a.
# And pow needs some care.

cuda10 = [
("add",".+","s+xi"),
("sub",".-","s-xi"),
("mul",".*","s*xi"),
("div","./","s/xi"),
("stoa","stoa","pow(s,xi)"),
("atos","atos","pow(xi,s)"),
("max","max","(xi>s?xi:s)"),
("min","min","(xi<s?xi:s)"),
("eq",".==","s==xi"),
("ne",".!=","s!=xi"),
("gt",".>","s>xi"),
("ge",".>=","s>=xi"),
("lt",".<","s<xi"),
("le",".<=","s<=xi"),
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
]

function cuda10def(f, j=f, o...)
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$(S)_10"
        @eval begin
            function $J(s::$T,x::KnetArray{$T})
                y = similar(x)
                ccall(($F,$libknet8),Void,(Cint,$T,Ptr{$T},Ptr{$T}),length(y),s,x,y)
                return y
            end
        end
    end
end

for f in cuda10
    if !isa(f,Tuple); f=(f,f); end
    j = Symbol(f[2])
    if isdefined(Base, j); eval(Expr(:import,:Base,j)); end
    cuda10def(f...)
end

# Additional imports
import Base: +, -, *, /, \, .^

# Ambiguity fixes:
max{T<:Real,S<:Real}(a::KnetArray{T},s::S)=max(T(s),a)
max{T<:Real,S<:Real}(s::S,a::KnetArray{T})=max(T(s),a)
min{T<:Real,S<:Real}(a::KnetArray{T},s::S)=min(T(s),a)
min{T<:Real,S<:Real}(s::S,a::KnetArray{T})=min(T(s),a)
max{T<:Real,S<:Number}(a::KnetArray{T},s::S)=max(T(s),a)
max{T<:Real,S<:Number}(s::S,a::KnetArray{T})=max(T(s),a)
min{T<:Real,S<:Number}(a::KnetArray{T},s::S)=min(T(s),a)
min{T<:Real,S<:Number}(s::S,a::KnetArray{T})=min(T(s),a)
(+)(a::KnetArray{Bool},s::Bool)=(.+)(s,a)
(+)(s::Bool,a::KnetArray{Bool})=(.+)(s,a)
(-)(a::KnetArray{Bool},s::Bool)=(.+)(-s,a)
(-)(s::Bool,a::KnetArray{Bool})=(.-)(s,a)
(.^)(x::Base.Irrational{:e}, a::KnetArray)=(.^)(float(x),a)

# For array,scalar we can get some for free:
# Only type corrected number,array need implementing for basic arithmetic:
(.+){T}(a::KnetArray{T},s::Number)=(.+)(T(s),a)
(.+){T}(s::Number,a::KnetArray{T})=(.+)(T(s),a)
(.-){T}(a::KnetArray{T},s::Number)=(.+)(T(-s),a)
(.-){T}(s::Number,a::KnetArray{T})=(.-)(T(s),a)
(.*){T}(a::KnetArray{T},s::Number)=(.*)(T(s),a)
(.*){T}(s::Number,a::KnetArray{T})=(.*)(T(s),a)
(./){T}(a::KnetArray{T},s::Number)=(.*)(T(1/s),a)
(./){T}(s::Number,a::KnetArray{T})=(./)(T(s),a)
max{T}(a::KnetArray{T},s::Number)=max(T(s),a)
max{T}(s::Number,a::KnetArray{T})=max(T(s),a)
min{T}(a::KnetArray{T},s::Number)=min(T(s),a)
min{T}(s::Number,a::KnetArray{T})=min(T(s),a)
(.^){T}(a::KnetArray{T},s::Number)=atos(T(s),a)
(.^){T}(s::Number,a::KnetArray{T})=stoa(T(s),a)

.=={T}(a::KnetArray{T},s::Number)=(T(s).==a)
.=={T}(s::Number,a::KnetArray{T})=(T(s).==a)
.!={T}(a::KnetArray{T},s::Number)=(T(s).!=a)
.!={T}(s::Number,a::KnetArray{T})=(T(s).!=a)
.>{T}(a::KnetArray{T},s::Number)=(T(s).<a)
.>{T}(s::Number,a::KnetArray{T})=(T(s).>a)
.>={T}(a::KnetArray{T},s::Number)=(T(s).<=a)
.>={T}(s::Number,a::KnetArray{T})=(T(s).>=a)
.<{T}(a::KnetArray{T},s::Number)=(T(s).>a)
.<{T}(s::Number,a::KnetArray{T})=(T(s).<a)
.<={T}(a::KnetArray{T},s::Number)=(T(s).>=a)
.<={T}(s::Number,a::KnetArray{T})=(T(s).<=a)

# familiar aliases for broadcasting operations of array & scalar (#7226):
(+){T}(a::KnetArray{T},s::Number)=(.+)(T(s),a)
(+){T}(s::Number,a::KnetArray{T})=(.+)(T(s),a)
(-){T}(a::KnetArray{T},s::Number)=(.+)(T(-s),a)
(-){T}(s::Number,a::KnetArray{T})=(.-)(T(s),a)
(*){T}(a::KnetArray{T},s::Number)=(.*)(T(s),a)
(*){T}(s::Number,a::KnetArray{T})=(.*)(T(s),a)
(/){T}(a::KnetArray{T},s::Number)=(.*)(T(1/s),a)
(\){T}(s::Number,a::KnetArray{T})=(.*)(T(1/s),a)

#(/){T}(s::Number,a::KnetArray{T})=(.*)(T(1/s),a) # not defined in base
#(^){T}(a::KnetArray{T},s::Number)=(.^)(a,T(s)) # linalg
#(^){T}(s::Number,a::KnetArray{T})=(.^)(T(s),a) # linalg
 
