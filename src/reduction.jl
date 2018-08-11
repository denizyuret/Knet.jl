# reduction.jl: Array->Scalar and Array->Vector reductions.

import Base: sum, prod, minimum, maximum

# countnz: deprecated reimplement using count(!iszero...)

# import AutoGrad: sumabs_, sumabs2_, minabs_, maxabs_
Base.sum(::typeof(abs), x::KnetArray, d...) = sumabs_(x,d...);
Base.sum(::typeof(abs2), x::KnetArray, d...) = sumabs2_(x,d...);
Base.maximum(::typeof(abs), x::KnetArray, d...) = maxabs_(x,d...);
Base.minimum(::typeof(abs), x::KnetArray, d...) = minabs_(x,d...);
reduced_dims_compat(dims,region)=map(last, Base.reduced_indices(map(Base.OneTo, dims), region))

reduction_ops = [
# The entry format is (cudaname, julianame, merge, item, init)
# ai is the accumulator, xi is the array element
("sum","sum","ai+xi","xi","0"),
("prod","prod","ai*xi","xi","1"),
("maximum","maximum","(ai>xi ? ai : xi)","xi","(-INFINITY)"),
("minimum","minimum","(ai<xi ? ai : xi)","xi","INFINITY"),
("sumabs","sumabs_","ai+xi","(xi<0 ? -xi : xi)","0"),
("sumabs2","sumabs2_","ai+xi","(xi*xi)","0"),
("maxabs","maxabs_","(ai>xi ? ai : xi)","(xi<0 ? -xi : xi)","0"),
("minabs","minabs_","(ai<xi ? ai : xi)","(xi<0 ? -xi : xi)","INFINITY"),
("countnz","countnz","ai+xi","(xi!=0)","0"),
]

function reduction_op(f, j=f, o...)
    J=Symbol(j)
    if isdefined(Base, J); eval(Expr(:import,:Base,J)); end
    for S in (32,64)
        T = Symbol("Float$S")
        F20 = "$(f)_$(S)_20"
        F21 = "$(f)_$(S)_21"
        F22 = "$(f)_$(S)_22"
        @eval begin
            # Array->Scalar reduction:
            function $J(x::KnetArray{$T})
                y=ccall(($F20,$libknet8),$T,(Cint,Ptr{$T}),length(x),x) # do not use @knet8, return not Nothing
                @gs; return y
            end
            # Array->Vector reduction:
            function $J(x::KnetArray{$T}, region)
                rdims = reduced_dims_compat(size(x), region)
                vdims = ndims(x)-length(region)
                if length(region) != 1 || ndims(x) == 1
                    vdims = count(x->x>1,rdims)
                end
                if vdims == 0   # falls back to Array->Scalar reduction
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
                    @knet8($F21,(Cint,Ptr{$T},Cint,Cint,Ptr{$T}),nx,x,sy,ny,y)
                    return y
                elseif vdims == ndims(x)-1
                    y = similar(x, rdims)
                    d = region[1]
                    nx = length(x); ny = length(y); s1 = stride(x,d)
                    s2 = stride(x,d+1); xd1 = size(x,d)-1
                    @knet8($F22,(Cint,Cint,Ptr{$T},Cint,Cint,Cint,Ptr{$T}),
                                 nx, xd1, x, s1, s2, ny, y)
                    return y
                else
                    y = $J(x,region[1])
                    f = $J==sumabs2 ? sum : $J
                    for k=2:length(region)
                        y = f(y,region[k])
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

@zerograd countnz(a,d...)

# Norm primitives:

# import Base.LinAlg: norm, vecnorm
using LinearAlgebra
import LinearAlgebra: norm

function norm(x::KnetArray{T}, p::Real=2) where {T}
    if length(x) == 0
        zero(T)
    elseif p == 2
        sqrt(sum(abs2,x))
    elseif p == 1
        sum(abs,x)
    elseif p == Inf
        maximum(abs,x)
    elseif p == 0
        countnz(x)
    elseif p == -Inf
        minimum(abs,x)
    else
        sum(abs.(x).^p)^(1/p)
    end
end

import Statistics: mean

mean(a::Union{T, Rec{T}}) where {T<:KnetArray} = sum(a) / length(a)
mean(a::Union{T, Rec{T}}, r) where {T<:KnetArray} = (b=sum(a,r); (b*convert(eltype(b),length(b)/length(a))))
mean(f::Function, a::Union{T, Rec{T}}) where {T<:KnetArray} = sum(f, a) / length(a)
