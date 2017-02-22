# broadcast.jl: Elementwise broadcasting binary functions for arrays and scalars.
# The entry format is (cudaname, julianame, kernelcode)
# With single name entries cudaname=julianame and kernelcode=name(xi,yi).
# I commented out functions if I don't know the Julia equivalent.

broadcast_ops = [
("add",".+","xi+yi"),
("sub",".-","xi-yi"),
("mul",".*","xi*yi"),
("div","./","xi/yi"),
("pow",".^","pow(xi,yi)"),
("max","max","(xi>yi?xi:yi)"),
("min","min","(xi<yi?xi:yi)"),
("eq",".==","xi==yi"),
("ne",".!=","xi!=yi"),
("gt",".>","xi>yi"),
("ge",".>=","xi>=yi"),
("lt",".<","xi<yi"),
("le",".<=","xi<=yi"),
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
("sigmback","sigmback","(xi*yi*(1-yi))"),
("tanhback","tanhback","(xi*(1-yi*yi))"),
("rpow","rpow","pow(yi,xi)"),   # need this for Array.^Scalar
]

# broadcast_op overloads a Julia function for KnetArrays.
# The corresponding kernel is defined in libknet8.
function broadcast_op(f, j=f, o...)
    J=Symbol(j)
    if isdefined(Base, J); eval(Expr(:import,:Base,J)); end
    for S in (32,64)
        T = Symbol("Float$S")
        F01 = "$(f)_$(S)_01"    # Scalar,Array->Array
        F11 = "$(f)_$(S)_11"    # Array,Array->Array (same size)
        F12 = "$(f)_$(S)_12"    # Array,Array->Array (different size)
        @eval begin
            function $J(x::$T,y::KnetArray{$T})
                z = similar(y)
                ccall(($F01,$libknet8),Void,(Cint,$T,Ptr{$T},Ptr{$T}),length(z),x,y,z)
                return z
            end
            function $J(x::KnetArray{$T},y::KnetArray{$T})
                if size(x)==size(y)
                    z = similar(x)
                    ccall(($F11,$libknet8),Void,(Cint,$Ptr{$T},Ptr{$T},Ptr{$T}),length(z),x,y,z)
                    return z
                else
                    (dz,sx,nx,sy,ny) = vbroadcast_shape(x,y)
                    z = similar(x,dz)
                    ccall(($F12,$libknet8),Void,(Cint,$Ptr{$T},Cint,Cint,Ptr{$T},Cint,Cint,Ptr{$T}),length(z),x,sx,nx,y,sy,ny,z)
                    return z
                end
            end
        end
    end
end
    
# vbroadcast_shape computes index/offset arguments for a broadcasting kernel call.
function vbroadcast_shape(x,y)
    nz = max(ndims(x),ndims(y))
    dz = ones(Int,nz)
    xdims = ydims = xsame = ysame = xlast = ylast = 0; zlen = 1
    for i=1:nz
        if size(x,i) > 1
            xdims += 1; xlast = i
            dz[i] = size(x,i)
        end
        if size(y,i) > 1
            ydims += 1; ylast = i
            if dz[i] == 1
                dz[i] = size(y,i)
            else
                dz[i] == size(y,i) || throw(DimensionMismatch("arrays could not be broadcast to a common size"))
            end
        end
        xsame += (dz[i] == size(x,i))
        ysame += (dz[i] == size(y,i))
        zlen *= dz[i]
    end
    xsame == nz || xdims <= 1 || error("Only vector broadcasting supported")
    ysame == nz || ydims <= 1 || error("Only vector broadcasting supported")
    if xdims == 0
        sx = zlen; nx = 1
    elseif xdims == 1
        sx = prod(dz[1:xlast-1]); nx=dz[xlast]
    elseif xsame == nz
        sx = 1; nx=zlen
    else
        error("Broadcasting error")
    end
    if ydims == 0
        sy = zlen; ny = 1
    elseif ydims == 1
        sy = prod(dz[1:ylast-1]); ny=dz[ylast]
    elseif ysame == nz
        sy = 1; ny=zlen
    else
        error("Broadcasting error")
    end
    return (tuple(dz...), sx, nx, sy, ny)
end

# Define all overloaded Julia functions for KnetArrays:

for f in broadcast_ops
    if !isa(f,Tuple); f=(f,); end
    broadcast_op(f...)
end

# Additional imports: fns in broadcast_ops are imported in broadcast_op()
import Base: +, -, *, /, \

# Here we'll just define some functions that specifically do not have broadcasting.
(+){T}(x::KnetArray{T},y::KnetArray{T})=(size(x)==size(y)||throw(DimensionMismatch("$(map(size,(x,y)))"));(.+)(x,y))
(-){T}(x::KnetArray{T},y::KnetArray{T})=(size(x)==size(y)||throw(DimensionMismatch("$(map(size,(x,y)))"));(.-)(x,y))
#(*){T}(x::KnetArray{T},y::KnetArray{T})=(.*)(x,y) # This is matmul
#(/){T}(x::KnetArray{T},y::KnetArray{T})=(./)(x,y) # This is another linalg op

# Broadcast max/min haven't been defined in Base:
# max(a::Array,b::Array)=broadcast(max,a,b)
# min(a::Array,b::Array)=broadcast(min,a,b)
# tkelman: These two methods aren't necessary, and overwrite Base. You can get this behavior via max.(a,b), with @compat needed on 0.4.

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

# Scalar kernels are defined for scalar,array order only.
# For array,scalar we can get most for free.
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
(.^){T}(s::Number,a::KnetArray{T})=(.^)(T(s),a)
# Pow is the one exception, we need to define a separate kernel:
(.^){T}(a::KnetArray{T},s::Number)=rpow(T(s),a)

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
 
