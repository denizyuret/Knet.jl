# reduction.jl: Array->Scalar and Array->Vector reductions.
# uses reduction_ops from ops.jl

import Base: sum, prod, minimum, maximum # , countnz

sum(::typeof(abs), x::KnetArray; dims=:) = sumabs(x,dims=dims)
sum(::typeof(abs2), x::KnetArray; dims=:) = sumabs2(x,dims=dims)
sum(::typeof(!iszero), x::KnetArray; dims=:) = countnz(x,dims=dims)
maximum(::typeof(abs), x::KnetArray; dims=:) = maxabs(x,dims=dims)
minimum(::typeof(abs), x::KnetArray; dims=:) = minabs(x,dims=dims)
sumabs(x;dims=:)=sum(abs,x;dims=dims)
sumabs2(x;dims=:)=sum(abs2,x;dims=dims)
maxabs(x;dims=:)=maximum(abs,x;dims=dims)
minabs(x;dims=:)=minimum(abs,x;dims=dims)
countnz(x;dims=:)=sum(!iszero,x;dims=dims)

reduced_dims_compat(dims,region)=map(last, Base.reduced_indices(map(Base.OneTo, dims), region))

function reduction_op(f, j=f, o...)
    J=Symbol(j)
    M = which(@__MODULE__, J)
    for S in (32,64)
        T = Symbol("Float$S")
        F20 = "$(f)_$(S)_20"
        F21 = "$(f)_$(S)_21"
        F22 = "$(f)_$(S)_22"
        @eval begin
            function ($M).$J(x::KnetArray{$T}; dims=:)
                if dims == Colon()
                    y=@knet8r($F20,$T,(Csize_t,Ptr{$T}),length(x),x)
                    return y
                end
                rdims = reduced_dims_compat(size(x), dims)
                vdims = ndims(x)-length(dims)
                if length(dims) != 1 || ndims(x) == 1
                    vdims = count(x->x>1,rdims)
                end
                if vdims == 0   # falls back to Array->Scalar reduction
                    return fill!(similar(x,rdims), $J(x))
                elseif vdims == 1
                    i0 = 0
                    ysize = ntuple(ndims(x)) do i
                        if in(i,dims)
                            1
                        elseif i0>0
                            error("Bad dims $dims")
                        else
                            i0=i
                            size(x,i)
                        end
                    end
                    y = similar(x, ysize)
                    nx = length(x); ny = length(y); sy = stride(x,i0)
                    @knet8($F21,(Csize_t,Ptr{$T},Csize_t,Csize_t,Ptr{$T}),nx,x,sy,ny,y)
                    return y
                elseif vdims == ndims(x)-1
                    y = similar(x, rdims)
                    d = dims[1]
                    nx = length(x); ny = length(y); s1 = stride(x,d)
                    s2 = stride(x,d+1); xd1 = size(x,d)-1
                    @knet8($F22,(Csize_t,Csize_t,Ptr{$T},Csize_t,Csize_t,Csize_t,Ptr{$T}),
                                 nx, xd1, x, s1, s2, ny, y)
                    return y
                else
                    y = $J(x,dims=dims[1])
                    f = $J==sumabs2 ? sum : $J
                    for k=2:length(dims)
                        y = f(y,dims=dims[k])
                    end
                    return y
                end
            end
        end
    end
end

for f in reduction_ops
    if !isa(f,Tuple); f=(f,); end
    reduction_op(f...)
end

