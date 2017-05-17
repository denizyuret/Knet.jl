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
        F13_x_y = "$(f)_$(S)_13_x_y"    # M-Array,N-Array->M-Array (M(x,y,z,w,t...), N(1,1,1,w,1...))
        F13_y_x = "$(f)_$(S)_13_y_x"   # x_y for correct ordering for compare operations,(kernel expects vector as second one)
        # F14_x_y = "$(f)_$(S)_14_x_y"    # Array,Array->Array ((M(x,y,z,w,t...), N(w,1,1,1...))
        # F14_y_x = "$(f)_$(S)_14_y_x"    # x_y for correct ordering for compare operations,(kernel expects vector as second one)

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

                      #  broadcasting first dimension and broadcast dim more than 127 and bigger dims are bigger than 511
                      # if you change those numbers update tests as well
                      # firstdimFlag= (xdims==1 && xlast==1) || (ydims==1 && ylast==1)
                      #
                      # if firstdimFlag
                      #   flat_dimsize=((xdims==1) ? (length(y)/length(x)) : (ydims==1) ? (length(x)/length(y)): -1)
                      #   # if   100<=flat_dimsize<128, if 128<=flat_dimsize<512, if 512<=flat_dimsize
                      #   #then      first_dimsize>2048,        first_dimsize>512,  first_dimsize>100 should be
                      #   # true if x is a vector that satisfies requirements
                      #   xvectorFlag= (xdims==1) ? ((length(x)>=2048 && (100<=flat_dimsize<128 )) || (length(x)>=512 && (128<=flat_dimsize<512 )) || (length(x)>=100 && (512<=flat_dimsize ))) : false
                      #   yvectorFlag= (ydims==1) ? ((length(y)>=2048 && (100<=flat_dimsize<128 )) || (length(y)>=512 && (128<=flat_dimsize<512 )) || (length(y)>=100 && (512<=flat_dimsize ))) : false
                      # end
                      # if (firstdimFlag && (xvectorFlag || yvectorFlag))
                      #       if (xdims==1)
                      #         # x is vector to be broadcasted,
                      #         @knet8($F14_y_x,(Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint),y,x,z,length(x),length(y),flat_dimsize)
                      #       else
                      #         @knet8($F14_x_y,(Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint),x,y,z,length(y),length(x),flat_dimsize)
                      #       end
                        # TODO-enis, broadcasting one element array might have done faster, like scalar to array broadcast
                        # if it is just one element, or broadcasting first dimension(or broadcast stride less than 512) ,or broadcast dimsize small than 704,call old-kernel
                        # if you change those numbers update tests as well
                        #half_BLOCK_SIZE_y=16
                        # n_block_13 = (B_N+(BLOCK_SIZE_y/2)-1)/(BLOCK_SIZE_y/2);
                        # if div((ny+15),16)<45
                        # div(brdcastdimstride,64)<8)
                      if (nx==1 || ny==1 || ((xdims==1 && (xlast==1 || 512<sx )) || (ydims==1 && (ylast==1 || sy<512 ))) || (xdims==1 && nx<704) || (ydims==1 && (ny<704)))
                            @knet8($F12,(Cint,Ptr{$T},Cint,Cint,Ptr{$T},Cint,Cint,Ptr{$T}),length(z),x,sx,nx,y,sy,ny,z)
                        # Array,Array->Array (M(x,y,z,w,t...), N(1,1,1,w,1...))

                      else
                            # x is vector to be broadcasted, then xlast is broadcasted dim
                            if (xdims==1)
                                brdcastdimstride = strides(y)[xlast]
                                # if broadcast dim is last dimension, nextstride is zero
                                brdcastnextstride = ((xlast+1) > ndims(y) ? 0: strides(y)[xlast+1])
                                multidimsize = prod(size(y)[xlast+1:end])
                                @knet8($F13_y_x,(Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint),y,x,z,brdcastdimstride,brdcastnextstride,multidimsize,length(y),length(x))
                            # y is vector to be broadcasted, then ylast is broadcasted dim
                            elseif (ydims==1)
                                brdcastdimstride = strides(x)[ylast]
                                # if broadcast last dimension, nextstride is zero
                                brdcastnextstride = ((ylast+1) > ndims(x) ? 0: strides(x)[ylast+1])
                                multidimsize = prod(size(x)[ylast+1:end])
                                @knet8($F13_x_y,(Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint),x,y,z,brdcastdimstride,brdcastnextstride,multidimsize,length(x),length(y))
                            else
                                error("Broadcasting error,caused by new kernel setup")
                            end
                        end
                    # multi dimensional broadcast
                    else
                        # dimcount_z=ndims(z);
                        stride_x=collect(Int32,strides(x));
                        stride_y=collect(Int32,strides(y));
                        stride_z=collect(Int32,strides(z));
                        dims_x=size(x)
                        dims_y=size(y)
                        # set broadcast dim strides of x and y to zero
                        # if they are not same and if dimsize is 1 then broadcast dim
                        for i in 1:ndims(x)
                            if dims_x[i]!=dims_y[i]
                                if dims_x[i]==1
                                    stride_x[i]=0
                                else
                                    stride_y[i]=0
                                end
                            end
                        end

                        if ndims(z)>5
                            stride_x=KnetArray(stride_x);
                            stride_y=KnetArray(stride_y);
                            stride_z=KnetArray(stride_z);

                            @knet8($F17,(Ptr{$T},Ptr{$T},Ptr{$T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint),x,y,z, stride_x, stride_y,stride_z, length(z), ndims(z))
                        else
                            # each kernel name ends with dimension count of result array
                            fname=Expr(:tuple,string($F16,"_",ndims(z)),:libknet8)
                            types=Expr(:tuple,Ptr{$T},Ptr{$T},Ptr{$T},ntuple(i->Cint,ndims(z)*3+1)...)
                            expr=Expr(:ccall,fname,Void,types,x,y,z,stride_x..., stride_y..., stride_z..., length(z))
                            eval(expr)
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
    multi=false
    if (xsame != nz && xdims > 1) || (ysame != nz && ydims > 1)
        multi = true
    end

    #x is only one element array
    if xdims == 0
        sx = zlen; nx = 1
    #x is a vector than sx=1 because xlast-1 is zero
    elseif xdims == 1
        #in that case xlast is broadcasting dim
        #so sx is stride of broadcasting dim in the z
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
