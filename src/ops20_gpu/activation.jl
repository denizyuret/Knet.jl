import Knet.Ops20: elu, relu, selu, sigm, eluback, reluback, seluback, sigmback
import Base.Broadcast: broadcasted
import Knet
using Knet.KnetArrays: KnetArray, DevArray, Bcasted
using Knet.LibKnet8: @knet8
using CUDA: CUDA, CuArray, CuPtr
using AutoGrad: AutoGrad, @primitive


# Specialize tanh gradient for DevArrays here. The others have been declared in Ops20 as generic gradients.
tanhback(dyi::T,yi::T) where {T<:Number} = dyi*(T(1)-yi*yi)
@primitive tanh(x::DevArray),dy,y tanhback.(dy,y)
@primitive tanhback(dy,y),ddx  ddx.*(1 .- y.*y)  ddx.*(-2 .* dy.*y)


for (R,P) in ((KnetArray,Ptr), (CuArray,CuPtr)), T in (Float32,Float64); S = sizeof(T) * 8
    for f in ("elu","relu","selu","sigm")
        J, F = Symbol(f), "$(f)_$S"; M = which(@__MODULE__,J)
        @eval begin
            function broadcasted(::typeof($J),x::$R{$T})
                y = similar(x)
                @knet8($F,(Cint,$P{$T},$P{$T}),length(y),x,y)
                return y
            end
            # Bcasted methods -- only needed for KnetArray
            ($M).$J(x::Bcasted{<:$R{$T}}) = broadcasted($J, x.value) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{<:$R{$T}}) = broadcasted($J, x.value) |> Bcasted
        end
    end
    for f in ("eluback","reluback","seluback","sigmback","tanhback")
        J, F = Symbol(f), "$(f)_$(S)_11"; M = which(@__MODULE__,J)
        @eval begin
            function broadcasted(::typeof($J),x::$R{$T},y::$R{$T})
                z = similar(x)
                @knet8($F,(Cint,$P{$T},$P{$T},$P{$T}),length(z),x,y,z)
                return z
            end
            # Bcasted methods -- only needed for KnetArray
            ($M).$J(x::Bcasted{<:$R{$T}}, y::Bcasted{<:$R{$T}}) = broadcasted($J, x.value, y.value) |> Bcasted
            ($M).$J(x::$R{$T}, y::Bcasted{<:$R{$T}}) = broadcasted($J, x, y.value) |> Bcasted
            ($M).$J(x::Bcasted{<:$R{$T}}, y::$R{$T}) = broadcasted($J, x.value, y) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{<:$R{$T}}, y::Bcasted{<:$R{$T}}) = broadcasted($J, x.value, y.value) |> Bcasted
            broadcasted(::typeof($J),x::$R{$T}, y::Bcasted{<:$R{$T}}) = broadcasted($J, x, y.value) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{<:$R{$T}}, y::$R{$T}) = broadcasted($J, x.value, y) |> Bcasted
        end
    end
end

