# binary.jl: Elementwise broadcasting binary functions for arrays and scalars.
# uses binary_ops from broadcast.jl.

import Base.Broadcast: broadcasted
import ..Ops20: eluback, reluback, seluback, sigmback

# binary_op defines the broadcast_func of a Julia function for KnetArrays.
# The corresponding kernel is defined in libknet8.
function binary_op(f, j=f, o...)
    J=Symbol(j)
    M = which(@__MODULE__, J)
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
            function broadcasted(::typeof($J),x::$T,y::KnetArray{$T})
                z = similar(y)
                @knet8($F01,(Cint,$T,Ptr{$T},Ptr{$T}),length(z),x,y,z)
                return z
            end
            # Array,Array->Array
            function broadcasted(::typeof($J),x::KnetArray{$T},y::KnetArray{$T})
                if size(x)==size(y)
                    z = similar(x)
                    @knet8($F11,(Cint,Ptr{$T},Ptr{$T},Ptr{$T}),length(z),x,y,z)
                    return z
                else
                    bs = vbroadcast_shape(x,y)
                    z = similar(x,bs[1])
                    _broadcasted($J,x,y,z,bs)
                end
            end
            # Helpers
            function _broadcasted(::typeof($J),x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,1},bs)
                if length(x) == 1
                    broadcasted($J,x[1],y)
                elseif length(y) == 1
                    broadcasted($J,x,y[1])
                else # length(x) == length(y) was handled above
                    throw(DimensionMismatch("$(map(size,(x,y,z)))"))
                end
            end
            function _broadcasted(::typeof($J),x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,2},bs)
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
            function _broadcasted(::typeof($J),x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,3},bs)
                sx,sy,sz = get_strides(x,y,z)
                @knet8($F16_3,
                       (Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                       x,y,z, sx[1],sx[2],sx[3], sy[1],sy[2],sy[3], sz[1],sz[2],sz[3], length(z))
                return z
            end
            function _broadcasted(::typeof($J),x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,4},bs)
                sx,sy,sz = get_strides(x,y,z)
                @knet8($F16_4,
                       (Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                       x,y,z, sx[1],sx[2],sx[3],sx[4], sy[1],sy[2],sy[3],sy[4], sz[1],sz[2],sz[3],sz[4], length(z))
                return z
            end
            function _broadcasted(::typeof($J),x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T,5},bs)
                sx,sy,sz = get_strides(x,y,z)
                @knet8($F16_5,
                       (Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                       x,y,z, sx[1],sx[2],sx[3],sx[4],sx[5], sy[1],sy[2],sy[3],sy[4],sy[5], sz[1],sz[2],sz[3],sz[4],sz[5], length(z))
                return z
            end
            function _broadcasted(::typeof($J),x::KnetArray{$T},y::KnetArray{$T},z::KnetArray{$T},bs)
                # ndims(z) <= 5 handled above, this is for > 5
                sx,sy,sz = map(s->convert(KnetArray, s), get_strides(x,y,z))
                @knet8($F17,
                       (Ptr{$T},Ptr{$T},Ptr{$T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint),
                       x,y,z, sx, sy, sz, length(z), ndims(z))
                return z
            end

            # Bcasted methods
            ($M).$J(x::Bcasted{<:KnetArray{$T}}, y::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x.value, y.value) |> Bcasted
            ($M).$J(x::KnetArray{$T}, y::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x, y.value) |> Bcasted
            ($M).$J(x::Bcasted{<:KnetArray{$T}}, y::KnetArray{$T}) = broadcasted($J, x.value, y) |> Bcasted
            ($M).$J(x::Bcasted{$T}, y::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x.value, y.value) |> Bcasted
            ($M).$J(x::$T, y::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x, y.value) |> Bcasted
            ($M).$J(x::Bcasted{$T}, y::KnetArray{$T}) = broadcasted($J, x.value, y) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{<:KnetArray{$T}}, y::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x.value, y.value) |> Bcasted
            broadcasted(::typeof($J),x::KnetArray{$T}, y::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x, y.value) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{<:KnetArray{$T}}, y::KnetArray{$T}) = broadcasted($J, x.value, y) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{$T}, y::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x.value, y.value) |> Bcasted
            broadcasted(::typeof($J),x::$T, y::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x, y.value) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{$T}, y::KnetArray{$T}) = broadcasted($J, x.value, y) |> Bcasted
        end # @eval
    end # for
    @eval begin # so we do not trigger some default Base implementation 
        ($M).$J(x::Bcasted, y::Bcasted) = throw(MethodError($J,(x,y)))
        ($M).$J(x, y::Bcasted) = throw(MethodError($J,(x,y)))
        ($M).$J(x::Bcasted, y) = throw(MethodError($J,(x,y)))
        broadcasted(::typeof($J),x::Bcasted, y::Bcasted) = throw(MethodError(broadcasted, ($J, x, y)))
        broadcasted(::typeof($J),x, y::Bcasted) = throw(MethodError(broadcasted, ($J, x, y)))
        broadcasted(::typeof($J),x::Bcasted, y) = throw(MethodError(broadcasted, ($J, x, y)))
    end
end # function binary_op

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
    # x,y,z may have different ndims, work with the max
    n = max(ndims(x), ndims(y), ndims(z))
    stride_x = Int32[ stride(x,i) for i=1:n ]
    stride_y = Int32[ stride(y,i) for i=1:n ]
    stride_z = Int32[ stride(z,i) for i=1:n ]
    dims_x = Int32[ size(x,i) for i=1:n ]
    dims_y = Int32[ size(y,i) for i=1:n ]
    for i in 1:n
        dims_x[i] == dims_y[i] && continue
        if dims_x[i]==1
            stride_x[i]=0
        else
            stride_y[i]=0
        end
    end
    return stride_x, stride_y, stride_z
end

# Additional imports: fns in binary_ops are defined using broadcasted.
import Base: +, -, *, /, \

# Here we'll just define some functions that specifically do not have broadcasting.
(+)(x::KnetArray{T},y::KnetArray{T}) where {T} = (size(x)==size(y)||throw(DimensionMismatch("$(map(size,(x,y)))"));(.+)(x,y))
(-)(x::KnetArray{T},y::KnetArray{T}) where {T} = (size(x)==size(y)||throw(DimensionMismatch("$(map(size,(x,y)))"));(.-)(x,y))
#(*){T}(x::KnetArray{T},y::KnetArray{T})=(.*)(x,y) # This is matmul
#(/){T}(x::KnetArray{T},y::KnetArray{T})=(./)(x,y) # This is another linalg op

# Broadcast max/min haven't been defined in Base:
# max(a::Array,b::Array)=broadcast(max,a,b)
# min(a::Array,b::Array)=broadcast(min,a,b)
# tkelman: These two methods aren't necessary, and overwrite Base. You can get this behavior via max.(a,b), with @compat needed on 0.4.

# import Base: broadcast

# Scalar kernels are defined for scalar,array order only.
# For array,scalar we can get most for free.

broadcasted(::typeof(+),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (.+)(T(s),a)
broadcasted(::typeof(+),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.+)(T(s),a)
broadcasted(::typeof(-),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (.+)(T(-s),a)
broadcasted(::typeof(-),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.-)(T(s),a)
broadcasted(::typeof(*),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (.*)(T(s),a)
broadcasted(::typeof(*),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.*)(T(s),a)
broadcasted(::typeof(/),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (.*)(T(1/s),a)
broadcasted(::typeof(/),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (./)(T(s),a)
broadcasted(::typeof(max),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = max.(T(s),a)
broadcasted(::typeof(max),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = max.(T(s),a)
broadcasted(::typeof(min),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = min.(T(s),a)
broadcasted(::typeof(min),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = min.(T(s),a)

# ^ does not work with cuda, trying to solve in CUDA.jl (JuliaGPU/CuArrays.jl#108)
# broadcasted(::typeof(^),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.^)(T(s),a)
# Pow is the one exception, we need to define a separate kernel:
# rpow(s,a)=a^s # only broadcast#rpow is defined above, we need rpow defined
# broadcasted(::typeof(^),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = rpow.(T(s),a)


broadcasted(::typeof(==),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (T(s).==a)
broadcasted(::typeof(==),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (T(s).==a)
broadcasted(::typeof(!=),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (T(s).!=a)
broadcasted(::typeof(!=),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (T(s).!=a)
broadcasted(::typeof(>),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (T(s).<a)
broadcasted(::typeof(>),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (T(s).>a)
broadcasted(::typeof(>=),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (T(s).<=a)
broadcasted(::typeof(>=),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (T(s).>=a)
broadcasted(::typeof(<),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (T(s).>a)
broadcasted(::typeof(<),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (T(s).<a)
broadcasted(::typeof(<=),a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (T(s).>=a)
broadcasted(::typeof(<=),s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (T(s).<=a)

# Bcasted methods

for f in Symbol.((+,-,*,/,max,min,^,==,!=,>,>=,<,<=))
    M = which(@__MODULE__, f)
    @eval begin
        broadcasted(::typeof($f),s::Bcasted{<:Number},a::Bcasted{KnetArray{T,N}}) where {T<:AbstractFloat,N} = broadcasted($f, s.value, a.value) |> Bcasted
        broadcasted(::typeof($f),s::Bcasted{<:Number},a::KnetArray{T,N}) where {T<:AbstractFloat,N} = broadcasted($f, s.value, a) |> Bcasted
        broadcasted(::typeof($f),s::Number,a::Bcasted{KnetArray{T,N}}) where {T<:AbstractFloat,N} = broadcasted($f, s, a.value) |> Bcasted
        broadcasted(::typeof($f),a::Bcasted{KnetArray{T,N}},s::Bcasted{<:Number}) where {T<:AbstractFloat,N} = broadcasted($f, a.value, s.value) |> Bcasted
        broadcasted(::typeof($f),a::KnetArray{T,N},s::Bcasted{<:Number}) where {T<:AbstractFloat,N} = broadcasted($f, a, s.value) |> Bcasted
        broadcasted(::typeof($f),a::Bcasted{KnetArray{T,N}},s::Number) where {T<:AbstractFloat,N} = broadcasted($f, a.value, s) |> Bcasted
        ($M).$f(s::Bcasted{<:Number},a::Bcasted{KnetArray{T,N}}) where {T<:AbstractFloat,N} = broadcasted($f, s.value, a.value) |> Bcasted
        ($M).$f(s::Bcasted{<:Number},a::KnetArray{T,N}) where {T<:AbstractFloat,N} = broadcasted($f, s.value, a) |> Bcasted
        ($M).$f(s::Number,a::Bcasted{KnetArray{T,N}}) where {T<:AbstractFloat,N} = broadcasted($f, s, a.value) |> Bcasted
        ($M).$f(a::Bcasted{KnetArray{T,N}},s::Bcasted{<:Number}) where {T<:AbstractFloat,N} = broadcasted($f, a.value, s.value) |> Bcasted
        ($M).$f(a::KnetArray{T,N},s::Bcasted{<:Number}) where {T<:AbstractFloat,N} = broadcasted($f, a, s.value) |> Bcasted
        ($M).$f(a::Bcasted{KnetArray{T,N}},s::Number) where {T<:AbstractFloat,N} = broadcasted($f, a.value, s) |> Bcasted
    end
end

# familiar aliases for broadcasting operations of array & scalar (#7226):
# (+)(a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (.+)(T(s),a)  -- deprecated
# (+)(s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.+)(T(s),a)  -- deprecated
# (-)(a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (.+)(T(-s),a) -- deprecated
# (-)(s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.-)(T(s),a)  -- deprecated
(*)(a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (.*)(T(s),a)
(*)(s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.*)(T(s),a)
(/)(a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (.*)(T(1/s),a)
(\)(s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.*)(T(1/s),a)

#(/)(s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.*)(T(1/s),a) # TODO: non-elementwise definition in linalg
#(^)(a::KnetArray{T},s::Number) where {T<:AbstractFloat} = (.^)(a,T(s)) # non-elementwise definition in linalg
#(^)(s::Number,a::KnetArray{T}) where {T<:AbstractFloat} = (.^)(T(s),a) # non-elementwise definition in linalg

tanhback(dyi::T,yi::T) where {T<:Number} = dyi*(T(1)-yi*yi)
@primitive tanh(x::KnetArray),dy,y tanhback.(dy,y)
@primitive tanhback(dy,y),ddx  ddx.*(1 .- y.*y)  ddx.*(-2 .* dy.*y)

# Define all overloaded Julia functions for KnetArrays:

for f in binary_ops
    if !isa(f,Tuple); f=(f,); end
    binary_op(f...)
end

# Fix #412 where KnetArray(randn(Float64,4,4,4,4)).^2 gives a 1-D result
broadcasted(::typeof(Base.literal_pow), ::typeof(^), k::KnetArray{T}, n::Val{N}) where {T,N} = broadcasted(^, k, N)
