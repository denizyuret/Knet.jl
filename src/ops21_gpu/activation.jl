import Knet.Ops21: elu, gelu, relu, selu, sigm, swish, tanh_
import Knet.Ops21: eluback, geluback, reluback, seluback, sigmback, swishback, tanh_back
import Base.Broadcast: broadcasted
import Knet
using Knet.KnetArrays: KnetArray, Bcasted
using CUDA: CUDA, CuArray, CuPtr
using Knet.LibKnet8: @knet8


for (A,P) in ((KnetArray,Ptr), (CuArray,CuPtr)), T in (Float32,Float64); S = sizeof(T) * 8
    for f in ("elu", "gelu", "selu", "sigm", "swish", "tanh_")
        J, Jback = Symbol(f), Symbol("$(f)back")
        M, Mback = which(@__MODULE__,J), which(@__MODULE__,Jback)
        F, Fback = "$(f)_$S", "$(f)back_$(S)_111"
        @eval begin

            function ($M).$J(x::$A{$T})
                y = similar(x)
                @knet8($F,(Cint,$P{$T},$P{$T}),length(y),x,y)
                return y
            end

            function ($Mback).$Jback(x::X, y::X, dy::X) where {X<:$A{$T}}
                dx = similar(x)
                @knet8($Fback,(Cint,$P{$T},$P{$T},$P{$T},$P{$T}),length(dx),x,y,dy,dx)
                return dx
            end
        end
    end

    for f in ("relu",)
        J, Jback = Symbol(f), Symbol("$(f)back")
        M, Mback = which(@__MODULE__,J), which(@__MODULE__,Jback)
        F, Fback = "$(f)_$(S)_1", "$(f)back_$(S)_1"
        @eval begin

            function ($M).$J(x::$A{$T}; max_value=Inf, negative_slope=0, threshold=0)
                y = similar(x)
                @knet8($F,(Cint,$T,$T,$T,$P{$T},$P{$T}),length(y),max_value,negative_slope,threshold,x,y)
                return y
            end

            function ($Mback).$Jback(x::X, y::X, dy::X; max_value=Inf, negative_slope=0, threshold=0) where {X<:$A{$T}}
                dx = similar(x)
                @knet8($Fback,(Cint,$T,$T,$T,$P{$T},$P{$T},$P{$T},$P{$T}),length(dx),max_value,negative_slope,threshold,x,y,dy,dx)
                return dx
            end
        end
    end
end

