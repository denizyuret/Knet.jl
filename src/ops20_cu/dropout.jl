import Knet.Ops20: dropout!, dropback!
using Knet.KnetArrays: @knet8, KnetArray
using CUDA: CuArray, CuPtr, seed!
using Random: rand!

# GPU implementation
for (R,P) in ((CuArray,CuPtr),(KnetArray,Ptr)), T in (Float16, Float32, Float64)
    S = sizeof(T)*8
    forw = Symbol("dropout_$S")
    back = Symbol("dropback_$S")
    @eval begin
        function dropout!(p::Number, x::$R{$T}, y::$R{$T}; seed=0)
            if seed !== nothing; CUDA.seed!(seed); end
            rand!(y)
            @knet8($forw,(Cint,$T,$P{$T},$P{$T}),length(y),$T(p),x,y)
            return y
        end
        function dropback!(p::Number, x::$R{$T}, y::$R{$T}, dy::$R{$T}, dx::$R{$T})
            @knet8($back,(Cint,$T,$P{$T},$P{$T},$P{$T},$P{$T}),length(dx),$T(p),x,y,dy,dx)
            return dx
        end
    end
end
