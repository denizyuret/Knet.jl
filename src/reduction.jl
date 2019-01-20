# reduction.jl: Array->Scalar and Array->Vector reductions.
# uses reduction_ops from ops.jl

import Base: sum, prod, minimum, maximum # , countnz
import LinearAlgebra: norm, lmul!
import Statistics: mean

sum(::typeof(abs), x::KnetArray; dims=:) = sumabs(x,dims=dims);
sum(::typeof(abs2), x::KnetArray; dims=:) = sumabs2(x,dims=dims);
maximum(::typeof(abs), x::KnetArray; dims=:) = maxabs(x,dims=dims);
minimum(::typeof(abs), x::KnetArray; dims=:) = minabs(x,dims=dims);
sumabs(x;dims=:)=sum(abs,x;dims=dims)
sumabs2(x;dims=:)=sum(abs2,x;dims=dims)
maxabs(x;dims=:)=maximum(abs,x;dims=dims)
minabs(x;dims=:)=minimum(abs,x;dims=dims)

reduced_dims_compat(dims,region)=map(last, Base.reduced_indices(map(Base.OneTo, dims), region))

function reduction_op(f, j=f, o...)
    J=Symbol(j)
    if isdefined(Base, J); eval(Expr(:import,:Base,J)); end
    for S in (32,64)
        T = Symbol("Float$S")
        F20 = "$(f)_$(S)_20"
        F21 = "$(f)_$(S)_21"
        F22 = "$(f)_$(S)_22"
        @eval begin
            function $J(x::KnetArray{$T}; dims=:)
                if dims == Colon()
                    y=@knet8r($F20,$T,(Cint,Ptr{$T}),length(x),x)
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
                    @knet8($F21,(Cint,Ptr{$T},Cint,Cint,Ptr{$T}),nx,x,sy,ny,y)
                    return y
                elseif vdims == ndims(x)-1
                    y = similar(x, rdims)
                    d = dims[1]
                    nx = length(x); ny = length(y); s1 = stride(x,d)
                    s2 = stride(x,d+1); xd1 = size(x,d)-1
                    @knet8($F22,(Cint,Cint,Ptr{$T},Cint,Cint,Cint,Ptr{$T}),
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

mean(a::Union{T, AutoGrad.Value{T}}; dims=:) where {T<:KnetArray} = (b=sum(a,dims=dims); b .* convert(eltype(b),(length(b)/length(a))))
mean(f::Base.Callable, a::Union{T, AutoGrad.Value{T}}) where {T<:KnetArray} = sum(f, a) / length(a)
# fixing ambiguity with AutoGrad
mean(f::typeof(abs), a::Union{T, AutoGrad.Value{T}}) where {T<:KnetArray} = sum(f, a) / length(a)
mean(f::typeof(abs2), a::Union{T, AutoGrad.Value{T}}) where {T<:KnetArray} = sum(f, a) / length(a)
