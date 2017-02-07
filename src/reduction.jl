# reduction.jl: Array->Scalar and Array->Vector reductions.

# The entry format is (cudaname, julianame, merge, item, init)
# ai is the accumulator, xi is the array element
reduction_ops = [
("sum","sum","ai+xi","xi","0"),
("prod","prod","ai*xi","xi","1"),
("maximum","maximum","(ai>xi?ai:xi)","xi","(-INFINITY)"),
("minimum","minimum","(ai<xi?ai:xi)","xi","INFINITY"),
("sumabs","sumabs","ai+xi","(xi<0?-xi:xi)","0"),
("sumabs2","sumabs2","ai+xi","(xi*xi)","0"),
("countnz","countnz","ai+xi","(xi!=0)","0"),
]

function reduction_op(f, j=f, o...)
    J=Symbol(j)
    if isdefined(Base, J); eval(Expr(:import,:Base,J)); end
    for S in (32,64)
        T = Symbol("Float$S")
        F20 = "$(f)_$(S)_20"
        F21 = "$(f)_$(S)_21"
        @eval begin
            # Array->Scalar reduction:
            function $J(x::KnetArray{$T})
                ccall(($F20,$libknet8),$T,(Cint,Ptr{$T}),length(x),x)
            end
            # Array->Vector reduction:
            function $J(x::KnetArray{$T}, region)
                rdims = Base.reduced_dims(size(x), region)
                vdims = count(x->x>1,rdims)
                if rdims == size(x)
                    return copy(x)
                elseif vdims == 0
                    return fill!(similar(x,rdims), $J(x))
                elseif vdims == 1
                    i0 = 0
                    ysize = ntuple(ndims(x)) do i
                        if in(i,region)
                            1
                        elseif i0>0
                            error("Bad region $region")
                        else
                            i0=i
                            size(x,i)
                        end
                    end
                    y = similar(x, ysize)
                    nx = length(x); ny = length(y); sy = stride(x,i0)
                    ccall(($F21,$libknet8),Void,(Cint,Ptr{$T},Cint,Cint,Ptr{$T}),nx,x,sy,ny,y)
                    return y
                else
                    error("Only scalar and vector reductions supported: $((size(x),region))")
                end
            end
        end
    end
end

for f in reduction_ops
    if !isa(f,Tuple); f=(f,); end
    reduction_op(f...)
end

# Norm primitives:

import Base.LinAlg: norm, vecnorm

norm(x::KnetVector, p::Real=2) = vecnorm(x, p)

function vecnorm{T}(x::KnetArray{T}, p::Real=2)
    if length(x) == 0
        zero(T)
    elseif p == 2
        sqrt(sumabs2(x))
    elseif p == 1
        sumabs(x)
    elseif p == Inf
        maximum(abs(x))
    elseif p == 0
        countnz(x)
    elseif p == -Inf
        minimum(abs(x))
    else
        sum(abs(x).^p)^(1/p)
    end
end

# The xentloss interface is no good because of double normalization.

"""
xentloss(x, p [,dims])

Compute cross-entropy loss for unnormalized log probability estimates
x and normalized probabilities p normalizing over the given
dimensions.  By default normalization is over the whole array, dims=1
normalizes over the columns, dims=2 normalizes over the rows of a 2-D
array.

"""
function xentloss(x,p,d...)
    x = x .- maximum(x,d...)
    z = log(sum(exp(x),d...))
    return sum(p .* (x .- z)) / length(z)
end

function xentback(x,p,d...)
    x = x .- maximum(x,d...)
    x = exp(x)
    z = sum(x,d...)
    return x./z - p
end

@primitive xentloss(x,p,d...),dy,y  (dy.*xentback(x,p,d...))

"""
logsumexp(x,[dims]) computes log(sum(exp(x),dims)) in a numerically
stable manner.  `dims` is an optional argument, if not specified the
summation is over the whole x, otherwise the summation is performed
over the given dimensions.  In particular dims=1 sums columns of x and
dims=2 sums rows of x.
"""
function logsumexp(x,d...)
    xmax = maximum(x,d...)
    xmax + log(sum(exp(x .- xmax),d...))
end

@primitive logsumexp(x,d...),dy,y  (dy .* exp(x .- y))

