import Knet.Ops20: elu, relu, selu, sigm, eluback, reluback, seluback, sigmback
import Base.Broadcast: broadcasted
import Knet
using Knet.KnetArrays: KnetArray, Bcasted
using Knet.LibKnet8: @knet8
using CUDA: CuArray, CuPtr

for (R,P) in ((KnetArray,Ptr), (CuArray,CuPtr)), T in (Float32,Float64); S = sizeof(T) * 8; M = Knet.Ops20
    for f in ("elu", "relu", "selu", "sigm"); J, F = Symbol(f), "$(f)_$S"
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
    for f in ("eluback", "reluback", "seluback", "sigmback"); J, F = Symbol(f), "$(f)_$(S)_11"
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
