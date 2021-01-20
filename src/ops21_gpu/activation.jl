import Knet.Ops21: gelu, geluback
import Base.Broadcast: broadcasted
import Knet
using Knet.KnetArrays: KnetArray, Bcasted
using CUDA: CuArray, CuPtr
using Knet.LibKnet8: @knet8


for (R,P) in ((KnetArray,Ptr), (CuArray,CuPtr)), T in (Float32,Float64); S = sizeof(T) * 8
    for f in ("gelu",)
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
    for f in ("geluback",)
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

