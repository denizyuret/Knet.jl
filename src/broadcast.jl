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

# broadcast_op defines the broadcast_func of a Julia function for KnetArrays.
# The corresponding kernel is defined in libknet8.
function broadcast_op(f, j=f, o...)
    J=broadcast_func(j)
    # @show J
    # if isdefined(Base, J); eval(Expr(:import,:Base,J)); end
    for S in (32,64)
        T = Symbol("Float$S")

        F01 = "$(f)_$(S)_01" # Scalar,Array->Array
        F11 = "$(f)_$(S)_11" # Array,Array->Array (same size) (not broadcast)
        F12 = "$(f)_$(S)_12" # Array,Array->Array (one have to be vector)
        F13_x_y = "$(f)_$(S)_13_x_y" # e.g. (A(x,y,z,w,t...), B(1,1,1,w,1...))
        F13_y_x = "$(f)_$(S)_13_y_x" # different versions for efficiency
        # F14_x_y = "$(f)_$(S)_14_x_y" # e.g. (M(x,y,z,w,t...), N(w,1,1,1...)
        # F14_y_x = "$(f)_$(S)_14_y_x" # different versions for efficiency
        # F15 reserved for another kernel, eliminated later and combined with F16
        F16_3 = "$(f)_$(S)_16_3" # multi-dimensional bcast unrolled up to 5 dims
        F16_4 = "$(f)_$(S)_16_4" # multi-dimensional bcast unrolled up to 5 dims
        F16_5 = "$(f)_$(S)_16_5" # multi-dimensional bcast unrolled up to 5 dims
        F17 = "$(f)_$(S)_17" # multi-dimensional bcast with loops

        @eval begin
            # Scalar,Array->Array
            function $J(x::$T,y::KnetArray{$T})
                z = similar(y)
                @knet8($F01,(Cint,$T,Ptr{$T},Ptr{$T}),length(z),x,y,z)
                return z
            end
            function $J(x::KnetArray{$T},y::KnetArray{$T})
                if size(x)==size(y)
                    z = similar(x)
                    @knet8($F11,(Cint,Ptr{$T},Ptr{$T},Ptr{$T}),length(z),x,y,z)
                    return z
                else
                    bs = vbroadcast_shape(x,y)
                    z = similar(x,bs[1])
                    $J(x,y,z,bs)
                end
            end
            function $J(x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,1},bs)
                if length(x) == 1
                    $J(x[1],y)
                elseif length(y) == 1
                    $J(x,y[1])
                else # length(x) == length(y) was handled above
                    throw(DimensionMismatch("$(map(size,(x,y,z)))"))
                end
            end
            function $J(x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,2},bs)
                # xlast or ylast will be broadcasting dimension
                (dz,sx,nx,sy,ny,xlast,ylast,xdims,ydims,multi) = bs
                if (nx == 1
                    || ny == 1
                    || ((xdims == 1 && (xlast==1 || 512 < sx )) ||
                        (ydims == 1 && (ylast==1 || sy < 512 )))
                    || (xdims==1 && nx<704)
                    || (ydims==1 && (ny<704)))
                    @knet8($F12,
                           (Cint,Ptr{$T},Cint,Cint,Ptr{$T},Cint,Cint,Ptr{$T}),
                           length(z),x,sx,nx,y,sy,ny,z)
                elseif xdims == 1
                    dim_stride = strides(y)[xlast]
                    next_stride = (xlast+1) > ndims(y) ?
                        0 : strides(y)[xlast+1]
                    dim_size = prod(size(y)[xlast+1:end])
                    @knet8($F13_y_x,
                           (Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint),
                           y,x,z,dim_stride,next_stride,dim_size,
                           length(y),length(x))
                elseif ydims == 1
                    dim_stride = strides(x)[ylast]
                    next_stride = (ylast+1) > ndims(x) ?
                        0 : strides(x)[ylast+1]
                    dim_size = prod(size(x)[ylast+1:end])
                    @knet8($F13_x_y,
                           (Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint),
                           x,y,z,dim_stride,next_stride,dim_size,
                           length(x), length(y))
                else
                    throw(DimensionMismatch("$(map(size,(x,y,z)))"))
                end
                return z
            end
            function $J(x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,3},bs)
                sx,sy,sz = get_strides(x,y,z)
                @knet8($F16_3,
                       (Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                       x,y,z, sx[1],sx[2],sx[3], sy[1],sy[2],sy[3], sz[1],sz[2],sz[3], length(z))
                return z
            end
            function $J(x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,4},bs)
                sx,sy,sz = get_strides(x,y,z)
                @knet8($F16_4,
                       (Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                       x,y,z, sx[1],sx[2],sx[3],sx[4], sy[1],sy[2],sy[3],sy[4], sz[1],sz[2],sz[3],sz[4], length(z))
                return z
            end
            function $J(x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,5},bs)
                sx,sy,sz = get_strides(x,y,z)
                @knet8($F16_5,
                       (Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                       x,y,z, sx[1],sx[2],sx[3],sx[4],sx[5], sy[1],sy[2],sy[3],sy[4],sy[5], sz[1],sz[2],sz[3],sz[4],sz[5], length(z))
                return z
            end
            function $J(x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T},bs)
                # ndims(z) <= 5 handled above, this is for > 5
                sx,sy,sz = map(s->convert(KnetArray, s), get_strides(x,y,z))
                @knet8($F17,
                       (Ptr{$T},Ptr{$T},Ptr{$T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint),
                       x,y,z, sx, sy, sz, length(z), ndims(z))
                return z
            end
        end # @eval
    end # for
end # function broadcast_op

# vbroadcast_shape computes index/offset arguments for a broadcasting kernel call.
function vbroadcast_shape(x,y)
    nz = max(ndims(x),ndims(y))
    dz = ones(Int,nz)
    xdims = ydims = xsame = ysame = xlast = ylast = 0; zlen = 1;
    sx = sy = -1;
    nx = ny = -1;

    for i = 1:nz
        # xdims: number of xdims != 1
        # xlast: last index != 1

        if size(x,i) > 1
            xdims += 1; xlast = i
            dz[i] = size(x,i)
        end
        if size(y,i) > 1
            ydims += 1; ylast = i
            if dz[i] == 1
                dz[i] = size(y,i)
            elseif dz[i] != size(y,i)
                throw(DimensionMismatch(
                    "arrays could not be broadcast to a common size"))
            end
        end

        xsame += (dz[i] == size(x,i))
        ysame += (dz[i] == size(y,i))
        zlen *= dz[i]
    end

    multi = false
    if (xsame != nz && xdims > 1) || (ysame != nz && ydims > 1)
        multi = true
    end

    if xdims == 0
        sx = zlen; nx = 1
    elseif xdims == 1
        sx = prod(dz[1:xlast-1]); nx=dz[xlast]
    elseif xsame == nz
        sx = 1
        nx = zlen
    elseif !multi
        error("Broadcasting error")
    end

    if ydims == 0
        sy = zlen
        ny = 1
    elseif ydims == 1
        sy = prod(dz[1:ylast-1])
        ny = dz[ylast]
    elseif ysame == nz
        sy = 1
        ny = zlen
    elseif !multi
        error("Broadcasting error")
    end

    return (tuple(dz...), sx, nx, sy, ny,xlast,ylast,xdims,ydims,multi)
end

function get_strides(x,y,z)
    stride_x = collect(Int32,strides(x));
    stride_y = collect(Int32,strides(y));
    stride_z = collect(Int32,strides(z));
    dims_x = size(x)
    dims_y = size(y)

    for i in 1:ndims(x)
        dims_x[i] == dims_y[i] && continue
        if dims_x[i]==1
            stride_x[i]=0
        else
            stride_y[i]=0
        end
    end
    return stride_x, stride_y, stride_z
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
if VERSION < v"0.6.0"; @eval begin
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
end; end

import Base: broadcast

# Scalar kernels are defined for scalar,array order only.
# For array,scalar we can get most for free.
if VERSION < v"0.6.0"; @eval begin
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
end; else; @eval begin
    $(broadcast_func(+)){T}(a::KnetArray{T},s::Number)=(.+)(T(s),a)
    $(broadcast_func(+)){T}(s::Number,a::KnetArray{T})=(.+)(T(s),a)
    $(broadcast_func(-)){T}(a::KnetArray{T},s::Number)=(.+)(T(-s),a)
    $(broadcast_func(-)){T}(s::Number,a::KnetArray{T})=(.-)(T(s),a)
    $(broadcast_func(*)){T}(a::KnetArray{T},s::Number)=(.*)(T(s),a)
    $(broadcast_func(*)){T}(s::Number,a::KnetArray{T})=(.*)(T(s),a)
    $(broadcast_func(/)){T}(a::KnetArray{T},s::Number)=(.*)(T(1/s),a)
    $(broadcast_func(/)){T}(s::Number,a::KnetArray{T})=(./)(T(s),a)
    $(broadcast_func(max)){T}(a::KnetArray{T},s::Number)=max.(T(s),a)
    $(broadcast_func(max)){T}(s::Number,a::KnetArray{T})=max.(T(s),a)
    $(broadcast_func(min)){T}(a::KnetArray{T},s::Number)=min.(T(s),a)
    $(broadcast_func(min)){T}(s::Number,a::KnetArray{T})=min.(T(s),a)
    $(broadcast_func(^)){T}(s::Number,a::KnetArray{T})=(.^)(T(s),a)
    # Pow is the one exception, we need to define a separate kernel:
    rpow(s,a)=a^s # only broadcast#rpow is defined above, we need rpow defined
    $(broadcast_func(^)){T}(a::KnetArray{T},s::Number)=rpow.(T(s),a)
end; end

if VERSION < v"0.6.0"; @eval begin
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
end; else; @eval begin
    $(broadcast_func(==)){T}(a::KnetArray{T},s::Number)=(T(s).==a)
    $(broadcast_func(==)){T}(s::Number,a::KnetArray{T})=(T(s).==a)
    $(broadcast_func(!=)){T}(a::KnetArray{T},s::Number)=(T(s).!=a)
    $(broadcast_func(!=)){T}(s::Number,a::KnetArray{T})=(T(s).!=a)
    $(broadcast_func(>)){T}(a::KnetArray{T},s::Number)=(T(s).<a)
    $(broadcast_func(>)){T}(s::Number,a::KnetArray{T})=(T(s).>a)
    $(broadcast_func(>=)){T}(a::KnetArray{T},s::Number)=(T(s).<=a)
    $(broadcast_func(>=)){T}(s::Number,a::KnetArray{T})=(T(s).>=a)
    $(broadcast_func(<)){T}(a::KnetArray{T},s::Number)=(T(s).>a)
    $(broadcast_func(<)){T}(s::Number,a::KnetArray{T})=(T(s).<a)
    $(broadcast_func(<=)){T}(a::KnetArray{T},s::Number)=(T(s).>=a)
    $(broadcast_func(<=)){T}(s::Number,a::KnetArray{T})=(T(s).<=a)
end; end

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
#(^){T}(a::KnetArray{T},s::Number)=(.^)(a,T(s)) # non-elementwise definition in linalg
#(^){T}(s::Number,a::KnetArray{T})=(.^)(T(s),a) # non-elementwise definition in linalg
