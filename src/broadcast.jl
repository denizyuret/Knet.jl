# broadcast.jl: Elementwise broadcasting binary functions for arrays and scalars.
# The entry format is (cudaname, julianame, kernelcode)
# With single name entries cudaname=julianame and kernelcode=name(xi,yi).
# I commented out functions if I don't know the Julia equivalent.

#(TODO-enis) all kernels creating fix size of thread blocks,
# as much as I know that should affect performance for small data

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
        F11 = "$(f)_$(S)_11"    # Array,Array->Array (same size) (not broadcast)
        F12 = "$(f)_$(S)_12"    # Array,Array->Array (different size) (one have to be vector)
        F13 = "$(f)_$(S)_13"    # M-Array,N-Array->M-Array (M(x,y,z,w,t...), N(1,1,1,w,1...))
        F14 = "$(f)_$(S)_14"    # Array,Array->Array (M(w,t), N(w)) (matrix, column vector)
        # F15 reserved for another kernel, eliminated later and combined with F16
        # loop unrolling (up to $unroll=ten dimensions)
        F16 = "$(f)_$(S)_16"    # Array,Array->Array (Multi dimensional broadcast)
        # for loop used, stride arrays passed
        F17 = "$(f)_$(S)_17"    # Array,Array->Array (Multi dimensional broadcast)

        @eval begin
            # Scalar,Array->Array
            function $J(x::$T,y::KnetArray{$T})
                z = similar(y)
                @knet8($F01,(Cint,$T,Ptr{$T},Ptr{$T}),length(z),x,y,z)
                return z
            end
            function $J(x::KnetArray{$T},y::KnetArray{$T})
                # Array,Array->Array (same size) (not broadcast)
                if size(x)==size(y)
                    z = similar(x)
                    @knet8($F11,(Cint,Ptr{$T},Ptr{$T},Ptr{$T}),length(z),x,y,z)
                    return z
                else
                    # xlast or ylast will be broadcasting dimension
                    (dz,sx,nx,sy,ny,xlast,ylast,xdims,ydims,multi) = vbroadcast_shape(x,y)
                    z = similar(x,dz)
                    # if it is not multi dimension broadcast, that can be applied vector oprimisations
                    if !multi
                        # for one dim array to matrix broadcast, 447 for good performance
                        if ((ndims(x)==2 && ndims(y)==1 && length(y)>447 )||(ndims(x)==1 && length(x)>447 && ndims(y)==2  ) )
                            if (ndims(x)==2)
                              println("col broadcast1")
                              @knet8($F14,(Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint),x,y,z,size(x,1),length(x))
                            else
                              println("col broadcast2")
                              @knet8($F14,(Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint),y,x,z,size(y,1),length(y))
                            end
                        # TODO-enis, broadcasting one element array might have done faster, like scalar to array broadcast
                        # if it one have just one element, or broadcasting first dimension,or broadcast dimsize small than 448,call old-kernel
                        elseif (nx==1 || ny==1 || ((xdims==1 && xlast==1) || (ydims==1 && ylast==1)) || (xdims==1 && nx<448) || (ydims==1 && ny<448))
                            # println("F122 xlast:$xlast ylast:$ylast dz $dz,sx $sx,nx $nx,sy $sy,ny $ny,xdims $xdims,ydims $ydims ylast:$ylast xlast:$xlast")
                            # println("ylast:$ylast xlast:$xlast")
                            @knet8($F12,(Cint,Ptr{$T},Cint,Cint,Ptr{$T},Cint,Cint,Ptr{$T}),length(z),x,sx,nx,y,sy,ny,z)
                        # Array,Array->Array (M(x,y,z,w,t...), N(1,1,1,w,1...))

                        else
                            # x is vector to be broadcasted, then xlast is broadcasted dim
                            if (xdims==1)
                                brdcastdimstride = strides(y)[xlast]
                                # if broadcast dim is last dimension, nextstride is zero
                                brdcastnextstride = ((xlast+1) > ndims(y) ? 0: strides(y)[xlast+1])
                                multidimsize = prod(size(y)[xlast+1:end])
                                # inputs to kernel explained in src/cuda13.jl
                                @knet8($F13,(Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint),y,x,z,brdcastdimstride,brdcastnextstride,multidimsize,length(x))
                            # y is vector to be broadcasted, then ylast is broadcasted dim
                            elseif (ydims==1)
                                brdcastdimstride = strides(x)[ylast]
                                # if broadcast last dimension, nextstride is zero
                                brdcastnextstride = ((ylast+1) > ndims(x) ? 0: strides(x)[ylast+1])
                                multidimsize = prod(size(x)[ylast+1:end])
                                @knet8($F13,(Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint),x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,length(y))
                            else
                                error("Broadcasting error,caused by new kernel setup")
                            end

                        end
                    # multi dimensional broadcast
                    else
                        dimcount_z=ndims(z);

                        if dimcount_z>10
                            stride_x=collect(Int32,strides(x));
                            stride_y=collect(Int32,strides(y));
                            stride_z=collect(Int32,strides(z));
                            dims_x=size(x)
                            dims_y=size(y)
                            # set broadcast dim strides of x and y to zero
                            # if they are not same and if dimsize is 1 then broadcast dim
                            for i in eachindex(size(x))
                                if dims_x[i]!=dims_y[i]
                                    if dims_x[i]==1
                                        stride_x[i]=0
                                    else
                                        stride_y[i]=0
                                    end
                                end
                            end
                            lz=length(z);nmz=ndims(z);
                            @knet8($F17,(Ptr{$T},Ptr{$T},Ptr{$T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint),x,y,z, stride_x, stride_y,stride_z, length(z), ndims(z))

                          else
                              # each kernel name ends with dimension count of result array
                              F16=string($F16,"_$dimcount_z")
                              # kernel input_1 will be Cint as many as dim count
                              kernel_input_1=",Cint"^(dimcount_z*3)
                              # delete the first comma
                              kernel_input_1=kernel_input_1[2:end]


                              stride_x=collect(Int32,strides(x));
                              stride_y=collect(Int32,strides(y));
                              stride_z=collect(Int32,strides(z));
                              kernel_input_2=""
                              for i=1:dimcount_z
                                kernel_input_2= string(kernel_input_2,",$stride_x[$i]")
                                kernel_input_2= string(kernel_input_2,",$stride_y[$i]")
                                kernel_input_2= string(kernel_input_2,",$stride_z[$i]")
                              end
                              kernel_input_2=kernel_input_2[2:end]
                              # kernel_input_2= string(kernel_input_2,",length(z), ndims(z)")

                              @knet8($F17,(Ptr{$T},Ptr{$T},Ptr{$T},eval(parse(kernel_input_1))...),x,y,z,eval(parse(kernel_input_2))...,length(z),ndims(z));
                          end
                      end


                    return z
                end
            end
        end
    end
end

# TODO-enis, rewrite vbroadcast_shape or a similar but more general function
# so that kernel branching is cleaner

# vbroadcast_shape computes index/offset arguments for a broadcasting kernel call.
function vbroadcast_shape(x,y)
    nz = max(ndims(x),ndims(y))
    dz = ones(Int,nz)
    xdims = ydims = xsame = ysame = xlast = ylast = 0; zlen = 1;
    # define for multi=true case
    sx=sy=-1;nx=ny=-1;
    # for each dimension
    for i=1:nz
        # xdims count dims of x whose size are not 1, and xlast is last index which is not one
        # and dz is filled with biggest dim sizes from each array
        #  if size of a dim in X bigger than 1
        if size(x,i) > 1
            xdims += 1; xlast = i
            dz[i] = size(x,i)
        end
        if size(y,i) > 1
            ydims += 1; ylast = i
            # if x is 1 in that dim than no problem for broadcast
            if dz[i] == 1
                dz[i] = size(y,i)
            else
                #  if also ydim is not 1 at the same position than cannot broadcast
                dz[i] == size(y,i) || throw(DimensionMismatch("arrays could not be broadcast to a common size"))
            end
        end
        # x-ysame counts how many dimsize of z same with theirs
        xsame += (dz[i] == size(x,i))
        ysame += (dz[i] == size(y,i))
        zlen *= dz[i]
    end
    # now all broadcast dims are supported
    # set $multi True, if other vector optimised kernels cannot be used
    # xsame == nz || xdims <= 1 || error("Only vector broadcasting supported")
    # ysame == nz || ydims <= 1 || error("Only vector broadcasting supported")
    multi=false
    if (xsame != nz && xdims > 1) || (ysame != nz && ydims > 1)
        multi = true
    end

    #x is only one element array
    if xdims == 0
        sx = zlen; nx = 1
    #x is a vector
    elseif xdims == 1
        sx = prod(dz[1:xlast-1]); nx=dz[xlast]
    # x is the N dim array
    elseif xsame == nz
        sx = 1; nx=zlen
    else
        if !multi
          sx=-1
          error("Broadcasting error")
        end
    end
    if ydims == 0
        sy = zlen; ny = 1
    elseif ydims == 1
        sy = prod(dz[1:ylast-1]); ny=dz[ylast]
    elseif ysame == nz
        sy = 1; ny=zlen
    else
      if !multi
        error("Broadcasting error")
      end
    end
    return (tuple(dz...), sx, nx, sy, ny,xlast,ylast,xdims,ydims,multi)
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
