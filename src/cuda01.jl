import Base: .+, .-, .*, ./, .^, max, min, .==, .>, .>=, .<, .<=, +, -, *, /, \

cuda01 = [
("add",".+","s+xi"),
("sub",".-","s-xi"),
("mul",".*","s*xi"),
("div","./","s/xi"),
("pow",".^","pow(s,xi)"),
("max","max","(xi>s?xi:s)"),
("min","min","(xi<s?xi:s)"),
("eq",".==","xi==s"),
("gt",".>","xi>s"),
("ge",".>=","xi>=s"),
("lt",".<","xi<s"),
("le",".<=","xi<=s"),
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

# ambiguity fixes:
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
.^(x::Base.Irrational{:e}, a::KnetArray)=.^(float(x),a)

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
#(.^){T}(a::KnetArray{T},s::Number) # cannot convert to an s,a operation
(.^){T}(s::Number,a::KnetArray{T})=(.^)(T(s),a)
max{T}(a::KnetArray{T},s::Number)=max(T(s),a)
max{T}(s::Number,a::KnetArray{T})=max(T(s),a)
min{T}(a::KnetArray{T},s::Number)=min(T(s),a)
min{T}(s::Number,a::KnetArray{T})=min(T(s),a)

.=={T}(a::KnetArray{T},s::Number)=(T(s).==a)
.=={T}(s::Number,a::KnetArray{T})=(T(s).==a)
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

function cuda01def(f, j=f, o...)
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$(S)_01"
        @eval begin
            function $J(s::$T,x::KnetArray{$T})
                y = similar(x)
                ccall(($F,$libknet8),Void,(Cint,$T,Ptr{$T},Ptr{$T}),length(y),s,x,y)
                return y
            end
        end
    end
end
    
#if isdefined(:libknet8)
    for f in cuda01
        isa(f,Tuple) || (f=(f,))
        cuda01def(f...)
    end
#end
